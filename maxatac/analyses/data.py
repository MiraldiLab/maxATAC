import glob
import shutil
import os
import sys
import stat
import logging
import re
import subprocess
from maxatac.utilities.system_tools import get_dir, check_data_packages_installed, get_version
from maxatac.utilities.constants import V1_MODELS_WISHLIST_URL

# Release and pre-release versions (2.0.0, 2.0.0b1, 2.0.0rc1, 2.0.0.post1) have a data repository tag of the
# same name; development versions (2.0.0.dev0, local versions such as 2.0.0+g1234abc) do not
RELEASE_VERSION = re.compile(r"\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?")


def remote_ref_exists(data_repo, ref):
    """Check whether a git repository has a tag or branch with this name

    Args:
        data_repo (str): Git repository URL or path
        ref (str): Tag or branch name

    Returns:
        bool: True if the tag or branch exists
    """
    result = subprocess.run(["git", "ls-remote", "--tags", "--heads", data_repo, ref],
                            capture_output=True, text=True, check=True)

    return any(line.split("\t")[1] in (f"refs/tags/{ref}", f"refs/heads/{ref}")
               for line in result.stdout.splitlines())


def resolve_data_version(data_repo, data_version):
    """Pick the data repository tag or branch to install

    Args:
        data_repo (str): Git repository URL or path
        data_version (str): --data_version: a tag or branch, "latest", or None for the tag matching this
        quant-maxATAC version

    Returns:
        str: Tag or branch to clone, or None to clone the latest commit on the default branch
    """
    if data_version == "latest":
        return None

    if data_version:
        if not remote_ref_exists(data_repo, data_version):
            raise ValueError(f"{data_repo} has no tag or branch named {data_version}. "
                             "Use --data_version latest for the newest data.")

        return data_version

    version = get_version()

    if not RELEASE_VERSION.fullmatch(version):
        logging.warning(f"quant-maxATAC {version} is a development version: installing the latest data from "
                        f"{data_repo}. Use --data_version to install a specific release.")
        return None

    tag = f"v{version}"

    if not remote_ref_exists(data_repo, tag):
        logging.warning(f"{data_repo} has no {tag} tag matching quant-maxATAC {version}: installing the latest "
                        "data instead. Use --data_version to install a specific release.")
        return None

    return tag


def remove_tree(path):
    """Delete a directory tree, including read-only files such as git objects on Windows

    Args:
        path (str): Directory to delete
    """
    def make_writable_and_retry(func, failed_path, _):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable_and_retry)
    else:
        shutil.rmtree(path, onerror=make_writable_and_retry)


def describe_existing_data(data_dir):
    """Summarize an existing maxATAC data directory so the user knows what will be overwritten

    Args:
        data_dir (str): Path to the installed data directory. Default: ~/opt/maxatac/data

    Returns:
        str: A short human readable description
    """
    lines = [f"Existing maxATAC data found in: {data_dir}"]

    if os.path.isdir(os.path.join(data_dir, ".git")):
        source = subprocess.run(["git", "remote", "get-url", "origin"], cwd=data_dir,
                                capture_output=True, text=True).stdout.strip()

        if source:
            lines.append(f"  Installed from: {source}")

    model_files = glob.glob(os.path.join(data_dir, "models", "*", "*.h5"))

    if model_files:
        n_tfs = len({os.path.dirname(model_file) for model_file in model_files})

        # quant-maxATAC ships a <TF>_cross_celltype.tsv calibration table next to each model; v1 (binary) models do not
        n_v1 = len([model_file for model_file in model_files
                    if not glob.glob(os.path.join(os.path.dirname(model_file), "*_cross_celltype.tsv"))])

        lines.append(f"  Models: {len(model_files)} .h5 files for {n_tfs} TFs" +
                     (f", {n_v1} of which look like maxATAC v1 (binary) models" if n_v1 else ""))

    return "\n".join(lines)


def confirm_overwrite(data_dir, assume_yes):
    """Ask the user to confirm that an existing data directory, including any v1 models, will be replaced

    Args:
        data_dir (str): Path to the installed data directory
        assume_yes (bool): Skip the prompt (--yes)

    Returns:
        bool: True if the data directory may be overwritten
    """
    logging.warning(describe_existing_data(data_dir))

    if assume_yes:
        logging.info("--yes was set: overwriting without asking")
        return True

    if not sys.stdin.isatty():
        raise RuntimeError(f"{data_dir} already exists and would be overwritten, but there is no terminal to "
                           "confirm this. Re-run with --yes to overwrite it.")

    try:
        answer = input(f"\n{data_dir} will be overwritten, including any maxATAC v1 models it contains.\n"
                       f"quant-maxATAC does not use the v1 models; to ask for v1 support, see {V1_MODELS_WISHLIST_URL}\n"
                       "Continue? [y/N] ")

    except EOFError:
        # No answer could be read (e.g. stdin is a null device on Windows): treat it as "no"
        logging.error("No answer received. Re-run with --yes to overwrite without asking.")
        return False

    return answer.strip().lower() in ("y", "yes")


def run_data(args):
    """Download maxATAC reference data and the quant-maxATAC models

    Workflow overview

    1) Pick the data version: --data_version, or the data repository tag matching this quant-maxATAC version
    2) If a data directory already exists, ask the user to confirm it will be overwritten (v1 models included)
    3) Clone that version of the data repository into a temporary directory, so a failed clone leaves the old
       data in place
    4) Replace the old data directory with the new clone, keeping any downloaded .2bit files
    5) Download the UCSC .2bit genome sequence for each requested genome

    Args:
        args (Namespace): Arguments list for the reference genome, output directory, data repository, data version,
        --yes and the log level.
    """
    check_data_packages_installed()

    ############### Parameters ###############
    # Path to the directory where the project data should be installed. Default: ~/opt
    base_dir = get_dir(args.output)

    # Path to the maxATAC directory in the install directory. Default: ~/opt/maxatac
    maxatac_dir = get_dir(os.path.join(base_dir, "maxatac"))

    # Final data directory. Default: ~/opt/maxatac/data
    maxatac_final_dir = os.path.join(maxatac_dir, "data")

    # Temporary clone directory, swapped in for the final directory once the clone succeeded
    maxatac_tmp_dir = os.path.join(maxatac_dir, ".maxatac_data.tmp")

    # The old data directory is renamed here before the new one is moved in, and deleted last
    maxatac_old_dir = os.path.join(maxatac_dir, ".maxatac_data.old")

    # Set all genomes if selected
    genomes = ["hg38", "hg19", "mm10"] if args.genome == ['all'] else args.genome

    ############### Body ###############
    # Resolve the version first, so a bad --data_version fails before anything is asked or changed
    data_ref = resolve_data_version(args.data_repo, args.data_version)
    data_label = data_ref or "latest"

    if os.path.isdir(maxatac_final_dir) and os.listdir(maxatac_final_dir):
        if not confirm_overwrite(maxatac_final_dir, args.yes):
            logging.info("Aborted: nothing was changed.")
            return

    # Clear temporary directories left behind by an interrupted run
    for leftover_dir in (maxatac_tmp_dir, maxatac_old_dir):
        if os.path.exists(leftover_dir):
            remove_tree(leftover_dir)

    logging.info(f"Cloning {args.data_repo} ({data_label}) \n" +
                 f"Final data will be placed in: {maxatac_final_dir}")

    branch = ["--branch", data_ref] if data_ref else []

    try:
        # advice.detachedHead=false: cloning a tag is expected, not worth git's detached HEAD notice
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--depth", "1", *branch,
                        args.data_repo, maxatac_tmp_dir], check=True)

    except subprocess.CalledProcessError:
        if os.path.exists(maxatac_tmp_dir):
            remove_tree(maxatac_tmp_dir)
        raise

    if os.path.isdir(maxatac_final_dir):
        # The .2bit files are not part of the repository and are large: carry them over instead of re-downloading
        for twobit in glob.glob(os.path.join(maxatac_final_dir, "*", "*.2bit")):
            new_genome_dir = os.path.join(maxatac_tmp_dir, os.path.basename(os.path.dirname(twobit)))

            if os.path.isdir(new_genome_dir) and not os.path.exists(os.path.join(new_genome_dir, os.path.basename(twobit))):
                shutil.move(twobit, new_genome_dir)

        # Swap by renaming, so the data directory is never left half deleted
        shutil.move(maxatac_final_dir, maxatac_old_dir)

    shutil.move(maxatac_tmp_dir, maxatac_final_dir)

    if os.path.exists(maxatac_old_dir):
        remove_tree(maxatac_old_dir)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=maxatac_final_dir,
                            capture_output=True, text=True).stdout.strip()

    logging.info(f"Installed {args.data_repo} {data_label} at commit {commit}")

    for genome in genomes:
        genome_dir = os.path.join(maxatac_final_dir, genome)

        if not os.path.isdir(genome_dir):
            logging.warning(f"{args.data_repo} has no reference data for {genome}; skipping the {genome}.2bit download")
            continue

        if os.path.exists(os.path.join(genome_dir, f"{genome}.2bit")):
            logging.info(f"{genome}.2bit is already present in {genome_dir}")
            continue

        logging.info(f"Downloading {genome}.2bit into {genome_dir}")

        # Download under a temporary name so an interrupted download is not mistaken for a finished one
        subprocess.run(["wget", "--no-check-certificate",
                        f"https://hgdownload.cse.ucsc.edu/goldenpath/{genome}/bigZips/{genome}.2bit",
                        "-O", f"{genome}.2bit.part"],
                       cwd=genome_dir, check=True)

        os.replace(os.path.join(genome_dir, f"{genome}.2bit.part"), os.path.join(genome_dir, f"{genome}.2bit"))

    logging.info("Finished!")
