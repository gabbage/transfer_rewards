import functools
import itertools
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm

from helpers.nlp_engine import prepare_batch
from scorer.data_helper.json_reader import readArticleRefs, readSortedScores


def run_absolute_rank_evaluation(model, fields, session_dir, phi=None, device=torch.device("cpu"), test_ids=None):
    sorted_scores = readSortedScores()
    article_refs = readArticleRefs()
    articles_dict = {}

    for entry in tqdm(article_refs, desc='Prepare the articles'):
        articles_dict.update({entry['id']: entry['article']})

    # Prepare csv output files for the correlation metrics and network outputs
    corr_file = open(os.path.join(session_dir, 'rank_correlations.csv'), 'w')
    corr_file.write('article,is_test,spearmanr_rho,spearmanr_p,pearson,pearson_p,kendall_tau,kendall_p\n')
    ranks_file = open(os.path.join(session_dir, 'rank_predictions.csv'), 'w')
    ranks_file.write('article,summ_id,summ_score,pred_rank\n')

    # Array with correlation data
    corr_data = np.zeros((len(sorted_scores), 6))
    test_mask = np.zeros(len(sorted_scores), dtype=bool)

    # Bring model into evaluation mode
    model.eval()
    phi.eval()
    model.to(device)
    phi.to(device)

    for i, (article_id, scores_list) in tqdm(enumerate(sorted_scores.items()),
                                             desc='Generate absolute ranks per article'):
        # Register that the article belongs to the test set
        if test_ids is not None and article_id in test_ids:
            test_mask[i] = 1

        # Reset the samples and preferences list for each article
        # human_rank = [s['rank'] for s in scores_list]
        human_ranks = [s['scores']['overall'] for s in scores_list]
        summary_key = 'summary' if 'summary' in fields else 'summary_1'

        # Preprocess summaries/article, pack them into a batch and process (= convert) the batch to pytorch tensor
        summary_batch = [fields[summary_key].preprocess(s['sys_summ']) for s in scores_list]
        article_batch = [fields['article'].preprocess(articles_dict[article_id])] * len(summary_batch)
        summary_batch_tensor = fields[summary_key].process(summary_batch, device)
        article_batch_tensor = fields['article'].process(article_batch, device)
        x = {summary_key: summary_batch_tensor, 'article': article_batch_tensor}

        # Disable the gradient computation
        with torch.no_grad():
            y = model(x) if phi is None else model(phi(x))
            nn_ranks = list(np.squeeze(y.cpu().numpy()))

        for s, nn_rank in zip(scores_list, nn_ranks):
            ranks_file.write('{},{},{:.2f},{:.4f}\n'.format(article_id, s['summ_id'], s['scores']['overall'], nn_rank))

        if len(nn_ranks) == len(human_ranks) and len(nn_ranks) > 0:
            spearmanr_result = spearmanr(human_ranks, nn_ranks)
            pearsonr_result = pearsonr(human_ranks, nn_ranks)
            kendalltau_result = kendalltau(human_ranks, nn_ranks)
            corr_file.write('{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(article_id,
                                                                                       test_mask[i],
                                                                                       spearmanr_result[0],
                                                                                       spearmanr_result[1],
                                                                                       pearsonr_result[0],
                                                                                       pearsonr_result[1],
                                                                                       kendalltau_result[0],
                                                                                       kendalltau_result[1]))

            corr_data[i, :] = [spearmanr_result[0], pearsonr_result[0], kendalltau_result[0],
                               spearmanr_result[1], pearsonr_result[1], kendalltau_result[1]]
        else:
            logging.warning('Ranking lists for article {} have different lengths!'.format(article_id))

    # Calculate the mean over all articles
    train_mask = np.asarray(1 - test_mask, dtype=np.bool)
    metrics = ['spearman-rho', 'pearson-r', 'kendall-tau',
               'spearman-rho_p-value', 'pearson-r_p-value', 'kendall-tau_p-value']
    results = {}

    if sum(test_mask) > 0:
        corr_mean_test = np.nanmean(corr_data[test_mask, :], axis=0)
        logging.info('Correlation mean on test  data spearmanr/pearsonr/kendall: {}'.format(corr_mean_test))
        plot_ranking_correlation_histogram(corr_data[test_mask, :], session_dir, 'test')

        for metric_key, value in zip(metrics, list(corr_mean_test)):
            results.update({'test_{}'.format(metric_key): [value]})

    if sum(train_mask) > 0:
        corr_mean_train = np.nanmean(corr_data[train_mask, :], axis=0)
        logging.info('Correlation mean on train data spearmanr/pearsonr/kendall: {}'.format(corr_mean_train))
        plot_ranking_correlation_histogram(corr_data[train_mask, :], session_dir, 'train')

        for metric_key, value in zip(metrics, list(corr_mean_train)):
            results.update({'train_{}'.format(metric_key): [value]})

    corr_mean_all = np.nanmean(corr_data, axis=0)
    logging.info('Correlation mean on all   data spearmanr/pearsonr/kendall: {}'.format(corr_mean_all))
    plot_ranking_correlation_histogram(corr_data, session_dir)

    # Flush and close the file handlers
    corr_file.flush()
    corr_file.close()
    ranks_file.flush()
    ranks_file.close()

    return results


def run_pairwise_rank_evaluation(model, fields, session_dir, pairing_method=None, phi=None, device=torch.device('cpu'),
                                 test_ids=None):
    pairing_method = functools.partial(itertools.permutations, r=2) if pairing_method is None else pairing_method
    sorted_scores = readSortedScores()
    article_refs = readArticleRefs()
    articles_dict = {}

    for entry in tqdm(article_refs, desc='Prepare the articles'):
        articles_dict.update({entry['id']: entry['article']})

    article_pref_graphs = {}

    # Prepare csv output files for the correlation metrics and network outputs
    corr_file = open(os.path.join(session_dir, 'rank_correlations.csv'), 'w')
    corr_file.write('article,is_test,spearmanr_rho,spearmanr_p,pearson,pearson_p,kendall_tau,kendall_p\n')
    ranks_file = open(os.path.join(session_dir, 'rank_predictions.csv'), 'w')
    ranks_file.write('article,summ1_id,summ2_id,summ1_score,summ2_score,summ1_pred,summ2_pred\n')

    # Array with correlation data
    corr_data = np.zeros((len(sorted_scores), 6))
    test_mask = np.zeros(len(sorted_scores), dtype=bool)

    # Bring model into evaluation mode
    model.eval()
    phi.eval()
    model.to(device)
    phi.to(device)

    for i, (article_id, scores_list) in tqdm(enumerate(sorted_scores.items()),
                                             desc='Generating model preferences for every article'):
        # Skip article because it belongs either to train or val split, but not to test split
        if test_ids is not None and article_id in test_ids:
            test_mask[i] = 1

        # Create directed graphs for each article and add the summary ids as nodes
        if article_id not in article_pref_graphs:
            article_pref_graphs.update({article_id: nx.DiGraph()})

        article_pref_graphs[article_id].add_nodes_from([s['summ_id'] for s in scores_list])

        # Build a list of pairs (e.g. permutation or combinations from itertools)
        summ_pairs = list(pairing_method(scores_list))

        # Reset the samples and preferences list for each article
        samples = {'summary_1': [], 'summary_2': [], 'article': [], 'label': []}
        preferences = []

        for (s1, s2) in summ_pairs:
            # Skip if s1 and s2 are the same
            if s1['sys_summ'] == s2['sys_summ'] and s1['scores']['overall'] != s2['scores']['overall']:
                logging.warning('Found equal summaries (in terms of text) with different scores!')

            if s1['summ_id'] == s2['summ_id']:
                continue

            # Compare all other pairs with the model
            samples['summary_1'].append(fields['summary'].preprocess(s1['sys_summ']))
            samples['summary_2'].append(fields['summary'].preprocess(s2['sys_summ']))
            samples['article'].append(fields['article'].preprocess(articles_dict[article_id]))
            samples['label'].append(fields['label'].preprocess(1 if s1['scores']['overall'] > s2['scores']['overall'] else -1))

        batch = {}
        batch['summary'] = [fields['summary'].process(samples['summary_1'], device),
                            fields['summary'].process(samples['summary_2'], device)]
        batch['article'] = fields['article'].process(samples['article'], device)
        batch['label'] = fields['article'].process(samples['article'], device)
        batch['input_fields'] = {'summary': fields['summary'], 'article': fields['article']}
        batch['target_fields'] = {'label': fields['label']}

        with torch.no_grad():
            # Pick input fields and target field from batch
            x, _ = prepare_batch(batch)

            # Forward pass: feed the input through the model
            y_pred = model(x) if phi is None else model(phi(x))
            preferences.extend(list(y_pred.split(1, dim=0)))

        # Build a preference graph with networkx
        for pref, pair in zip(preferences, summ_pairs):
            p1, p2 = pair
            np_pref = torch.squeeze(pref).cpu().numpy()

            if len(np_pref.shape) == 1:
                a = np_pref[0]
                b = np_pref[1]
                pref_val = np.argmax(np_pref)
            elif len(np_pref.shape) == 0:
                a = np_pref.item()
                b = np_pref.item()
                pref_val = 0 if a > 0 else -1

            ranks_file.write('{},{},{},{:.2f},{:.2f},{:.4f},{:.4f}\n'.format(article_id, p1['summ_id'], p2['summ_id'],
                                                                             p1['scores']['overall'],
                                                                             p2['scores']['overall'], a, b))

            from_node = pair[0]['summ_id'] if pref_val == 0 else pair[1]['summ_id']
            to_node = pair[1]['summ_id'] if pref_val == 0 else pair[0]['summ_id']
            prev_weight = 0

            if article_pref_graphs[article_id].has_edge(from_node, to_node):
                prev_weight = article_pref_graphs[article_id].get_edge_data(from_node, to_node)['weight']

            article_pref_graphs[article_id].add_edge(from_node, to_node, weight=prev_weight + 1)

        # List the human summary rankings
        # ranking_human = [v['rank'] for v in sorted_scores[_id]]
        human_ranks = [s['scores']['overall'] for s in scores_list]

        if nx.is_directed_acyclic_graph(article_pref_graphs[article_id]):
            ranking_nn = [len(nx.descendants(article_pref_graphs[article_id], node)) for node in article_pref_graphs[article_id].nodes]
        else:
            logging.warning('Preference graph for article {} is not a directed acyclic graph!'.format(article_id))
            ranking_nn = shift_rank([len(article_pref_graphs[article_id].out_edges(node))
                                    for node in article_pref_graphs[article_id].nodes])

        if len(ranking_nn) == len(human_ranks) and len(ranking_nn) > 0:
            spearmanr_result = spearmanr(human_ranks, ranking_nn)
            pearsonr_result = pearsonr(human_ranks, ranking_nn)
            kendalltau_result = kendalltau(human_ranks, ranking_nn)
            corr_file.write('{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(article_id,
                                                                                       test_mask[i],
                                                                                       spearmanr_result[0],
                                                                                       spearmanr_result[1],
                                                                                       pearsonr_result[0],
                                                                                       pearsonr_result[1],
                                                                                       kendalltau_result[0],
                                                                                       kendalltau_result[1]))

            corr_data[i, :] = [spearmanr_result[0], pearsonr_result[0], kendalltau_result[0],
                               spearmanr_result[1], pearsonr_result[1], kendalltau_result[1]]
        else:
            logging.warning('Ranking lists for article {} have different lengths!'.format(article_id))

        # plot_pref_graph(article_pref_graphs, article_id, scores_list, session_dir)

    # Calculate the mean over all articles
    train_mask = np.asarray(1 - test_mask, dtype=np.bool)
    metrics = ['spearman-rho', 'pearson-r', 'kendall-tau',
               'spearman-rho_p-value', 'pearson-r_p-value', 'kendall-tau_p-value']
    results = {}

    if sum(test_mask) > 0:
        corr_mean_test = np.nanmean(corr_data[test_mask, :], axis=0)
        logging.info('Correlation mean on test  data spearmanr/pearsonr/kendall: {}'.format(corr_mean_test))
        plot_ranking_correlation_histogram(corr_data[test_mask, :], session_dir, 'test')

        for metric_key, value in zip(metrics, list(corr_mean_test)):
            results.update({'test_{}'.format(metric_key): [value]})

    if sum(train_mask) > 0:
        corr_mean_train = np.nanmean(corr_data[train_mask, :], axis=0)
        logging.info('Correlation mean on train data spearmanr/pearsonr/kendall: {}'.format(corr_mean_train))
        plot_ranking_correlation_histogram(corr_data[train_mask, :], session_dir, 'train')

        for metric_key, value in zip(metrics, list(corr_mean_train)):
            results.update({'train_{}'.format(metric_key): [value]})

    corr_mean_all = np.nanmean(corr_data, axis=0)
    logging.info('Correlation mean on all   data spearmanr/pearsonr/kendall: {}'.format(corr_mean_all))
    plot_ranking_correlation_histogram(corr_data, session_dir)

    # Flush and close the file handlers
    corr_file.flush()
    corr_file.close()
    ranks_file.flush()
    ranks_file.close()

    return results


def plot_pref_graph(graph, article_id, scores_list, session_dir=None):
    plt.ioff()
    fig = plt.figure()
    plt.title('Preference graph for article {}'.format(article_id))
    nx.draw_networkx(graph[article_id], labels={i['summ_id']: i['sys_name'] for i in scores_list},
                     pos=nx.circular_layout(graph[article_id]))

    if session_dir is not None:
        if not os.path.exists(os.path.join(session_dir, 'graphs')):
            os.makedirs(os.path.join(session_dir, 'graphs'))

        output_file = os.path.join(session_dir, 'graphs', 'graph_{}.pdf'.format(article_id))
    else:
        output_file = os.path.join('.pdfs', 'graph_{}.pdf'.format(article_id))

    plt.savefig(output_file)
    plt.show(block=False)
    plt.draw()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.close(fig)


def pref_graph_total_order(g, nodes):
    if len(nodes) == 0:
        return []
    elif len(nodes) == 1:
        return nodes
    else:
        pivot_node = np.random.choice(nodes)
        reachable_nodes = [to_node for (from_node, to_node) in g.out_edges(pivot_node) if to_node in nodes]
        other_nodes = (set(nodes) - {pivot_node}) - set(reachable_nodes)

        return pref_graph_total_order(g, list(other_nodes)) + [pivot_node] + pref_graph_total_order(g, list(
            reachable_nodes))


def total_order_to_rank(g, nodes):
    result = []
    rank = 0

    for i in range(len(nodes) - 1):
        result.append((nodes[i], rank))

        if not g.has_edge(nodes[i + 1], nodes[i]):
            rank += 1

    result.append((nodes[i + 1], rank))
    return result


def shift_rank(rank_list):
    return list(np.array(rank_list) - np.min(rank_list))


def plot_ranking_correlation_histogram(data, session_dir, split_key='all'):
    assert len(data.shape) == 2

    if data.shape[1] == 7:
        data = data[:, 1:]

    assert data.shape[1] in [3, 6]

    plt.ioff()
    fig, a = plt.subplots(data.shape[1] // 3, 3, figsize=(16, 8))
    a = a.ravel()

    plt.suptitle('Rank correlation metrics histogram on {} data'.format(split_key))
    labels = ['spearmanr_rho', 'pearson', 'kendall_tau', 'spearmanr_p', 'pearson_p', 'kendall_tau_p']
    bins = np.linspace(-1.0, 1.0, 20)

    for idx, ax in enumerate(a):
        data_col = data[:, idx]
        data_col = data_col[~np.isnan(data_col)]
        ax.hist(data_col, label=labels[idx], bins=bins, align='mid')

        # for i in bins:
        #     ax.axvline(x=i, color='black', linestyle=':', linewidth=1)
        mean = np.mean(data_col)
        std = np.std(data_col)
        ax.axvline(x=np.mean(data_col), color='orange', label='$\\mu$')
        ax.axvspan(mean - std, mean + std, alpha=0.25, color='orange', label='$\\mu \\pm \\sigma$')

        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel('value')
        ax.set_ylabel('# articles')
        ax.legend()
        ax.grid()

    plt.savefig(os.path.join(session_dir, 'rank_correlations_hist_{}.pdf'.format(split_key)), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_ranking_correlation_histogram(np.genfromtxt('ranking_results.csv', delimiter=',', skip_header=1), '')
    exit(0)

    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3, 4, 5])
    g.add_edges_from([(2, 3), (1, 4), (4, 5), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (1, 3)])  # , (2, 1), (1, 2)])

    # print(g.out_edges(1))
    print('Is dag? ', nx.is_directed_acyclic_graph(g))
    total_order = pref_graph_total_order(g, g.nodes)
    print('Total order: ', total_order)
    ranking = total_order_to_rank(g, total_order)
    print('Ranking: ', sorted(ranking, key=lambda x: x[0]))
    my_ranking = shift_rank([len(g.in_edges(node)) for node in g.nodes])
    print('Old ranking: ', list(zip(g.nodes, my_ranking)))

    nx.draw_networkx(g, pos=nx.circular_layout(g))
    plt.show()
