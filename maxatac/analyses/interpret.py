import os
from os import path
import keras
import pickle, h5py
import shap
import seaborn as sns
import modisco
import pandas as pd

from maxatac.utilities.training_tools import get_roi_pool

from maxatac.utilities.system_tools import get_dir
from maxatac.utilities.interpretation_tools import generating_interpret_data, dinuc_shuffle_DNA_only_several_times, \
    combine_DNA_only_mult_and_diffref, output_meme_pwm

from maxatac.utilities.constants import (
    BP_RESOLUTION,
)
from maxatac.utilities.training_tools import MaxATACModel

from collections import OrderedDict
from matplotlib import pyplot as plt

from modisco.visualization import viz_sequence
from modisco.tfmodisco_workflow import workflow
import modisco.affinitymat.core
import modisco.cluster.phenograph.core
import modisco.cluster.phenograph.cluster
import modisco.cluster.core
import modisco.aggregator
from modisco.util import *

# custom objects
from maxatac.architectures.dcnn import loss_function, dice_coef

keras_model_custom_objects_dict = {
    'loss_function': loss_function,
    'dice_coef': dice_coef
}


def run_interpretation(args):
    """
    Interpret a trained maxATAC models using TFmodisco.


    _________________
    Workflow Overview

    1) Initialize the maxATAC model
    2)

    :param args: arch, seed, output, prefix, threads, meta_file, quant, dense,
    weights, chroms, interpret_roi

    :return: A directory of results related to interpretation.
    """
    maxatac_model = MaxATACModel(interpret=True,
                                 interpret_cell_type="HepG2",
                                 arch=args.arch,
                                 seed=args.seed,
                                 output_directory=args.output,
                                 prefix=args.prefix,
                                 threads=args.threads,
                                 meta_path=args.meta_file,
                                 weights=args.weights
                                 )

    os.chdir(maxatac_model.interpret_location)

    maxatac_model.nn_model.layers[-2].activation = keras.activations.linear
    maxatac_model.nn_model.save(maxatac_model.interpret_model_file)

    # re-load model with custom functions
    keras.backend.clear_session()

    model = keras.models.load_model(maxatac_model.interpret_model_file,
                                    custom_objects={'loss_function': loss_function, 'dice_coef': dice_coef}
                                    )

    interpret_pool = get_roi_pool(filepath=args.interpret_roi, chroms=args.chroms, shuffle=True)

    print('Interpretation pool loaded, total size: ', interpret_pool.shape)

    print('training cell lines: ', maxatac_model.cell_types)

    interpret_pool_group = interpret_pool[interpret_pool['Chr'].isin(args.chroms)].reset_index(drop=True).groupby(
        ['Cell_Line', 'ROI_Type'])

    interpret_set = []

    for _count, _idx in enumerate(interpret_pool_group.groups):
        if _count < 2:  # use only one background cell line data for interpreting analysis
            _group = interpret_pool_group.get_group(_idx)
            _group.reset_index(drop=True, inplace=True)

            interpret_data_X_file = path.join(maxatac_model.interpret_location, 'X_{}_{}.pkl'.format(_idx[0], _idx[1]))
            interpret_data_y_file = path.join(maxatac_model.interpret_location, 'y_{}_{}.pkl'.format(_idx[0], _idx[1]))

            interpret_set.append([interpret_data_X_file, interpret_data_y_file, _idx[0], _idx[1]])

            print(_idx, _group.shape[0])

            X, y = generating_interpret_data(
                sequence=args.sequence,
                average=args.average,
                meta_table=maxatac_model.meta_dataframe,
                roi_pool=_group,
                train_cell_lines=maxatac_model.cell_types,
                rand_ratio=args.rand_ratio,
                train_tf=args.train_tf,
                tchroms=args.chroms,
                bp_resolution=BP_RESOLUTION,
                filters=None,
                workers=args.threads,
            )

            with open(interpret_data_X_file, 'wb') as f:
                pickle.dump(X, f)

            with open(interpret_data_y_file, 'wb') as f:
                pickle.dump(y, f)

    # interpreting section
    interpret_set_df = pd.DataFrame(interpret_set, columns=['X_file', 'y_file', 'Cell', 'PC_region'])
    grouped_interpret_set_df = interpret_set_df.groupby('Cell')
    for _count, _cell_idx in enumerate(grouped_interpret_set_df.groups):
        if _count == 0:
            _group = grouped_interpret_set_df.get_group(_cell_idx)
            # print(_group)

            if _group.shape[0] != 2:
                print('Cannot interpret more than 2 types of regions!')
                raise NotImplementedError

            else:
                _group_neg = _group[_group['PC_region'] == 'ATAC']
                _group_pos = _group[_group['PC_region'] == 'CHIP']
                # print(_group_pos['X_file'])
                # print(_group_pos['y_file'])

                with open(_group_pos['X_file'].values.tolist()[0], 'rb') as f:
                    X_pos = pickle.load(f)

                with open(_group_pos['y_file'].values.tolist()[0], 'rb') as f:
                    y_pos = pickle.load(f)

                # ATAC-seq
                with open(_group_neg['X_file'].values.tolist()[0], 'rb') as f:
                    X_neg = pickle.load(f)

                with open(_group_neg['y_file'].values.tolist()[0], 'rb') as f:
                    y_neg = pickle.load(f)

                # reformatting for X and y
                X_pos = [k for k in X_pos]
                X_pos = np.array(X_pos)

                y_pos = [k for k in y_pos]
                y_pos = np.array(y_pos)

                X_neg = [k for k in X_neg]
                X_neg = np.array(X_neg)

                y_neg = [k for k in y_neg]
                y_neg = np.array(y_neg)

                # check some basics about these labels
                plt.subplots(figsize=(10, 5))

                sns.barplot(y=np.mean(y_pos, axis=0), x=list(range(32)), color='blue', alpha=0.75,
                            label='ChIP-seq regions')
                sns.barplot(y=np.mean(y_neg, axis=0), x=list(range(32)), color='red', alpha=0.75,
                            label='ATAC-seq regions')

                plt.xlabel('channel #')

                plt.ylabel('mean target value')

                plt.legend()

                plt.savefig(os.path.join(maxatac_model.interpret_location,
                                         'Target_channel_distribition_{}_{}.pdf'.format(args.train_tf, _cell_idx)),
                            bbox_inches='tight')
                # sample a few from pos/neg pairs

                sample_size = np.minimum(args.interpret_sample, np.minimum(X_pos.shape[0], X_neg.shape[0]))

                pos_index = np.random.choice(range(X_pos.shape[0]), sample_size, replace=False)

                neg_index = np.random.choice(range(X_neg.shape[0]), sample_size, replace=False)

                X = np.concatenate((X_pos[pos_index, :, :],
                                    X_neg[neg_index, :, :]))

                y = np.concatenate((y_pos[pos_index, :],
                                    y_neg[neg_index, :]))

                with open(os.path.join(maxatac_model.interpret_location,
                                       'X_interpreting_samples_{}_{}.pkl'.format(args.train_tf, _cell_idx)), 'wb') as f:
                    pickle.dump(X, f)

                with open(os.path.join(maxatac_model.interpret_location,
                                       'y_interpreting_samples_{}_{}.pkl'.format(args.train_tf, _cell_idx)), 'wb') as f:
                    pickle.dump(y, f)

                # deeplift analysis using shap library
                task_list = args.interpret_channel_list

                task_list_labels = []

                contrib = OrderedDict()

                hypo_contrib = OrderedDict()

                for task in task_list:
                    task_list_labels.append("task" + str(task))
                    explainer = shap.DeepExplainer(model=([model.input], model.layers[-1].output[:, task]),
                                                   data=dinuc_shuffle_DNA_only_several_times,
                                                   combine_mult_and_diffref=combine_DNA_only_mult_and_diffref)

                    hypo_contrib[task] = explainer.shap_values(X)
                    contrib[task] = hypo_contrib[task] * X

                # save contribution scores
                f = h5py.File(
                    os.path.join(maxatac_model.interpret_location, "{}_{}_scores.h5".format(args.train_tf, _cell_idx)))

                g = f.create_group("contrib_scores")

                for task_idx in task_list:
                    g.create_dataset("task" + str(task_idx),
                                     data=contrib[task_idx][:, :, :4])  # chop off the ATAC-seq signal contributions

                g = f.create_group("hyp_contrib_scores")

                for task_idx in task_list:
                    g.create_dataset("task" + str(task_idx),
                                     data=hypo_contrib[task_idx][:, :,
                                          :4])  # chop off the ATAC-seq signal contributions
                # g = f.create_group("contrib_scores_FULL")
                # for task_idx in task_list:
                #     g.create_dataset("task" + str(task_idx) + '_FULL',
                #                      data=contrib[task_idx][:, :, :])
                # g = f.create_group("hyp_contrib_scores_FULL")
                # for task_idx in task_list:
                #     g.create_dataset("task" + str(task_idx)+'_FULL',
                #                      data=hypo_contrib[task_idx][:, :, :])

                # generate task to score mapping
                task_to_scores = OrderedDict()
                task_to_hyp_scores = OrderedDict()
                tasks = f["contrib_scores"].keys()

                for task in tasks:
                    # Note that the sequences can be of variable lengths;
                    # in this example they all have the same length (200bp) but that is
                    # not necessary.
                    task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task][:]]
                    task_to_hyp_scores[task] = [np.array(x) for x in f['hyp_contrib_scores'][task][:]]

                f.close()

                # tf modisco analysis
                null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)

                tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                    # Slight modifications from the default settings
                    sliding_window_size=15,
                    flank_size=5,
                    target_seqlet_fdr=0.15,
                    seqlets_to_patterns_factory=
                    modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                        trim_to_window_size=15,
                        initial_flank_to_add=5,
                        kmer_len=5, num_gaps=1,
                        num_mismatches=0,
                        final_min_cluster_size=60)
                )(task_names=task_list_labels,
                  contrib_scores=task_to_scores,
                  hypothetical_contribs=task_to_hyp_scores,
                  one_hot=X[:, :, :4],
                  null_per_pos_scores=null_per_pos_scores
                  )

                grp = h5py.File(os.path.join(maxatac_model.interpret_location,
                                             "tf_modsico_results_{}_{}.hdf5".format(args.train_tf, _cell_idx)))
                tfmodisco_results.save_hdf5(grp)

                grp.close()

                # estimated background from actual sequences
                background = np.sum(X[:, :, :4], axis=(0, 1)) / (X.shape[0] * X.shape[1])

                # modisco workflow
                track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
                    task_names=task_list_labels,
                    contrib_scores=task_to_scores,
                    hypothetical_contribs=task_to_hyp_scores,
                    one_hot=X[:, :, :4])

                grp = h5py.File(os.path.join(maxatac_model.interpret_location,
                                             "tf_modsico_results_{}_{}.hdf5".format(args.train_tf, _cell_idx)), "r")
                loaded_tfmodisco_results = workflow.TfModiscoResults.from_hdf5(grp, track_set=track_set)

                # generate cluster heatmap
                activity_patterns = np.array(grp['metaclustering_results']['attribute_vectors'])[
                    np.array(
                        [
                            x[0] for x in sorted(
                            enumerate(grp['metaclustering_results']['metacluster_indices']),
                            key=lambda x: x[1])
                        ]
                    )
                ]

                # clustered channel contribution patterns
                ax = sns.heatmap(activity_patterns, center=0)

                ax.figure.savefig(os.path.join(maxatac_model.interpret_location,
                                               '{}_{}_metacluster_heatmap.pdf'.format(args.train_tf, _cell_idx)),
                                  bbox_inches='tight')

                # get all metaclusters and generate trimmed patterns
                metacluster_names = [x.decode("utf-8") for x in
                                     list(grp["metaclustering_results"]["all_metacluster_names"][:])
                                     ]

                for metacluster_name in metacluster_names:
                    print(metacluster_name)

                    metacluster_grp = (grp["metacluster_idx_to_submetacluster_results"][metacluster_name])

                    print("activity pattern:", metacluster_grp["activity_pattern"][:])

                    all_pattern_names = [x.decode("utf-8") for x in
                                         list(metacluster_grp["seqlets_to_patterns_result"]["patterns"][
                                                  "all_pattern_names"][:])]

                    if len(all_pattern_names) == 0:
                        print("No motifs found for this activity pattern")

                    for pattern_idx, pattern_name in enumerate(all_pattern_names):
                        get_dir(path.join(maxatac_model.metacluster_patterns_location, args.train_tf, _cell_idx))
                        file_name = os.path.join(maxatac_model.metacluster_patterns_location, args.train_tf, _cell_idx,
                                                 metacluster_name + '.' + pattern_name + '.IC_trimmed')
                        # original pattern
                        untrimmed_pattern = (
                            loaded_tfmodisco_results
                                .metacluster_idx_to_submetacluster_results[metacluster_name]
                                .seqlets_to_patterns_result.patterns[pattern_idx])

                        # trimmed pattern
                        trimmed_pattern = untrimmed_pattern.trim_by_ic(ppm_track_name="sequence",
                                                                       background=background,
                                                                       threshold=0.3)
                        # save
                        viz_sequence.plot_weights_and_save(
                            viz_sequence.ic_scale(np.array(trimmed_pattern["sequence"].fwd),
                                                  background=background),
                            file_name=file_name
                        )
                        # meme pattern
                        # output IC trimmed, meme formatted pwm, along with update
                        output_pattern_name = os.path.join(maxatac_model.meme_query_pattern_location,
                                                           metacluster_name + '-' + pattern_name)
                        pattern_idx = get_ic_trimming_indices(np.array(untrimmed_pattern["sequence"].fwd),
                                                              background=background, threshold=0.3)
                        trimmed_pwm = np.array(untrimmed_pattern["sequence"].fwd)[pattern_idx[0]:pattern_idx[1], :]
                        output_meme_pwm(trimmed_pwm, output_pattern_name)

                pattern_seqlet_df = []

                for i in grp['metacluster_idx_to_submetacluster_results'].keys():
                    for j in grp['metacluster_idx_to_submetacluster_results'][i] \
                                     ['seqlets_to_patterns_result']['patterns'].keys() - ['all_pattern_names']:
                        for k in grp['metacluster_idx_to_submetacluster_results'][i] \
                                ['seqlets_to_patterns_result']['patterns'][j]['seqlets_and_alnmts']['seqlets']:
                            pattern_seqlet_df.append(
                                [int(k.decode().split(',')[0].strip('\'').split(':')[1]), k.decode(), i, j,
                                 list(grp['metacluster_idx_to_submetacluster_results'][i]['activity_pattern'])])

                pattern_seqlet_df = pd.DataFrame(pattern_seqlet_df,
                                                 columns=['sample index', 'seqlet id', 'metacluster id', 'pattern id',
                                                          'metacluster pattern'])
                pattern_seqlet_df.to_csv(os.path.join(maxatac_model.interpret_location,
                                                      '{}_{}_seqlet_sample_metacluster_pattern_info.csv'.format(
                                                          args.train_tf, _cell_idx)), header=True, index=True, sep=',')

                grp.close()
