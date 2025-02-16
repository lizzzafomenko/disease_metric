import pandas as pd
import collections as col
import itertools as it
import warnings
import numpy as np
warnings.simplefilter('ignore')
import seaborn as sbs
import matplotlib.pyplot as plt
import scipy.stats as sts
import random
random.seed(42)


class HPO:

    def __init__(self, is_a_file, genes_to_phen):
        print(f'Initializing (is_a: {is_a_file}, genes_to_phenotype: {genes_to_phen}): start')

        # create a DAG from HPO version
        self.merge, self.backs =  self._create(is_a_file)  # self.merge = v -> [merged_v]  # self.backs = v -> [parents_v]

        # OMIM gene to phenotypes
        self.phens = pd.read_csv(genes_to_phen, sep='\t')

        # toposort
        self.Vs = self._toposort()

        # all ancestors to root; number of children nodes (including itself)
        self.ancestors, self.number_of_children = self._paths()

        # count prevalence of phen within all diseases
        self.prevalence = self._phen_prevalence()

        print('Initializing: done')


    # create a DAG from hp-base.is_a.tsv

    def _create(self, is_a_file):
        merge = col.defaultdict(list)
        backs = col.defaultdict(list)
        with open(is_a_file, 'r') as file:
            file.readline()
            for line in file:
                term = line.split('\t')[0]
                parent = line.split('\t')[2]
                merge[parent].append(term)
                backs[term].append(parent)
        return [merge, backs]


    # classic toposort for a DAG

    def _toposort(self, start='HP:0000001'):
        order = []
        seen = []

        def deep(v, seen, order):
            seen.append(v)
            for m_v in self.merge[v]:
                if m_v not in seen:
                    deep(m_v, seen, order)
            order.append(v)

        # know that root = HP:0000001
        deep(start, seen, order)

        return order[::-1]


    # collect paths from phenotype to root

    def _paths(self):
        # df = pd.DataFrame(index=Vs, columns=Vs, data=0, dtype=float)
        paths = col.defaultdict(list)
        paths['HP:0000001'] = [[]]
        for term in self.Vs[1::]:
            for parent in self.backs[term]:
                for path in paths[parent]:
                    paths[term].append([parent]+path)


        for key, value in paths.items():
            paths[key] = set(it.chain.from_iterable(value))


        nnodes = col.defaultdict(int)
        for key, values in paths.items():
            for value in values:
                nnodes[value] += 1


        return [paths, nnodes]


    # '-' freqs will count as const you insert (default = 0.1)
    def frequency(self, const=False):

        if not const:
            print('fill missed values ("-") as mean')
        else:
            print(f'fill missed values ("-") as {const}')

        def assign_freq(x):
            if str(x).find('/') != -1:
                # Bayes: k / n -> k+1 / n+2
                return (int(x.split('/')[0]) + 1) / (int(x.split('/')[1]) + 2)
            elif str(x).find('%') != -1:
                return float(x.strip('%'))/100
            elif x == 'HP:0040280':
                return 1
            elif x == 'HP:0040281':
                return 0.9
            elif x == 'HP:0040282':
                return 0.5
            elif x == 'HP:0040283':
                return 0.175
            elif x == 'HP:0040284':
                return 0.02175
            elif x == 'HP:0040284':
                return 0
            if const:
                return const
            return x

        self.phens['frequency'] = [1 if self.phens['hpo_id'][i] in ['HP:0000007', 'HP:0000006'] else
                                   assign_freq(self.phens['frequency'][i]) for i in range(self.phens.shape[0])]

        # if not const assigned for - frequencies, we should count mean!
        if not const:
            a = self.phens[self.phens['frequency'] != '-'].groupby('hpo_id').frequency.mean().to_dict()
            total_mean = self.phens[self.phens['frequency'] != '-'].frequency.mean()

            def abs_freq(phen):
                if phen in a.keys():
                    return a[phen]
                else:
                    return total_mean

            self.phens['frequency'] = [abs_freq(self.phens['hpo_id'][i]) if self.phens['frequency'][i] == '-' else
                                       self.phens['frequency'][i] for i in range(self.phens.shape[0])]


        return self.phens


    # number of pairs (gene, dis) which are associated with phen

    def _phen_prevalence(self):
        # phenotype : [(Gene, Dis), (Gene, Dis), ...]
        dis_num = self.phens.groupby(['gene_symbol', 'disease_id']).size().shape[0]  # total number of pairs gene ~ OMIM:xxx in HPO (6886; unique OMIM ids 6240)
        dis_un = col.defaultdict(list, self.phens.groupby('hpo_id').apply(lambda g: list(zip(g['gene_symbol'], g['disease_id']))).to_dict())

        # phen_cnts = phens.groupby(['gene_symbol', 'disease_id']).hpo_id.nunique().to_dict()
        phen_prevalence = {}

        for term in self.Vs[::-1]:
            dis_set = []
            for next in self.merge[term]:
                dis_set += list(dis_un[next])
            dis_set += list(dis_un[term])
            dis_un[term] = list(dis_set)
            phen_prevalence[term] = len(set(dis_set))/dis_num


        return phen_prevalence




    def OMIMxHPO_1(self, cosine1):
        # create a df
        alls = self.phens.groupby(['gene_symbol', 'disease_id'])[['hpo_id', 'frequency']].apply(lambda g: g.values.tolist()).to_dict()

        df_1 = pd.DataFrame(data = 0, index = [f'{key[0]},{key[1]}' for key in alls.keys()], columns = self.Vs)

        print(f'starting the matrix OMIM x HPO where phen~dis freq is not used; shape - {df_1.shape[0]} pairs Genes~Dis and {df_1.shape[1]} phenotypes')
        total_nodes = len(self.Vs)

        abs_in_is_a = set()
        for key, value in alls.items():
            for pair in value:
                for anc in self.ancestors[pair[0]]:
                    if anc in self.Vs:
                        df_1.at[f'{key[0]},{key[1]}', anc] = (1 - self.prevalence[anc]) * (1 - self.number_of_children[anc] / total_nodes)
                    else:
                        abs_in_is_a.add(anc)
                if pair[0] in self.Vs:
                    df_1.at[f'{key[0]},{key[1]}', pair[0]] = (1 - self.prevalence[pair[0]]) * (1 - self.number_of_children[pair[0]] / total_nodes)
                else:
                    abs_in_is_a.add(pair[0])
        print(f'The following HP terms are absent in is_a file (check the versions!): {", ".join(abs_in_is_a)}')
        print(f'OMIM x HPO matrix done with shape {df_1.shape}')

        #df_1.to_csv(omimhpo1, sep='\t', index=True)

        # set 0 for phenotypes we assume as unusable
        df_1['HP:0000118'] = [0] * df_1.shape[0]
        g = self._toposort('HP:0000005')
        for t in g:
            df_1[t] = [0] * df_1.shape[0]

        dis_cos_1 = pd.DataFrame(data=1, columns=df_1.index, index=df_1.index)

        print(f'starting calculating the cosine matrix, {dis_cos_1.shape}')

        miniresult = pd.DataFrame(np.dot(df_1, np.transpose(df_1)), index=df_1.index, columns = df_1.index)

        l = 0
        for pair in it.combinations(df_1.index, r=2):
            cos = miniresult.at[pair[0], pair[1]] / (miniresult.at[pair[0], pair[0]] * miniresult.at[pair[1], pair[1]])**0.5
            dis_cos_1.loc[pair[0], pair[1]] = dis_cos_1.loc[pair[1], pair[0]] = cos
            l+=1
            if l % 100000 == 0:
                print(l // 100000, 'from', 237)


        dis_cos_1.to_csv(cosine1, sep='\t', index=True)
        print(f'cosine matrix saved to file {cosine1}')




    def OMIMxHPO_freq(self, cosinefreq):
        # create a df
        alls = self.phens.groupby(['gene_symbol', 'disease_id'])[['hpo_id', 'frequency']].apply(lambda g: g.values.tolist()).to_dict()


        df_freq = pd.DataFrame(data = 0, index = [f'{key[0]},{key[1]}' for key in alls.keys()], columns = self.Vs)

        total_nodes = len(self.Vs)



        print('calculating frequency for every Gene~Phen~Dis association...')
        phen_to_dis_freq = {}
        for key, value in alls.items():
            freqs = col.defaultdict(list)
            for pair in value:
                for anc in self.ancestors[pair[0]]:
                    freqs[anc].append(pair[1])
                freqs[pair[0]].append(pair[1])

            for phen, freq_set in freqs.items():
                if len(freq_set) == 1:
                    phen_to_dis_freq[(key[0], key[1], phen)] = freq_set[0]
                else:
                    k = 1
                    for i in freq_set:
                        k *= (1-i)
                    phen_to_dis_freq[(key[0], key[1], phen)] = 1-k

        print('frequencies: done')

        print(f'starting the matrix OMIM x HPO where phen~dis freq IS used; shape - {df_freq.shape[0]} pairs Genes~Dis and {df_freq.shape[1]} phenotypes')
        abs_in_is_a = set()
        for key, value in alls.items():
            for pair in value:
                for anc in self.ancestors[pair[0]]:
                    if anc in self.Vs:
                        df_freq.at[f'{key[0]},{key[1]}', anc] = (1 - self.prevalence[anc]) * (1 - self.number_of_children[anc] / total_nodes) * phen_to_dis_freq[(key[0], key[1], anc)]
                    else:
                        abs_in_is_a.add(anc)
                if pair[0] in self.Vs:
                    df_freq.at[f'{key[0]},{key[1]}', pair[0]] = (1 - self.prevalence[pair[0]]) * (1 - self.number_of_children[pair[0]] / total_nodes) * phen_to_dis_freq[(key[0], key[1], pair[0])]
                else:
                    abs_in_is_a.add(pair[0])
        print(f'The following HP terms are absent in is_a file (check the versions!): {", ".join(abs_in_is_a)}')
        print(f'OMIM x HPO matrix done with shape {df_freq.shape}')


        # set 0 for phenotypes we assume as unusable
        df_freq['HP:0000118'] = [0]*df_freq.shape[0]
        g = self._toposort('HP:0000005')
        for t in g:
            df_freq[t] = [0]*df_freq.shape[0]

        dis_cos_freq = pd.DataFrame(data=1, columns=df_freq.index, index=df_freq.index)
        print(f'start calculating the cosine matrix, {dis_cos_freq.shape}')

        miniresult = pd.DataFrame(np.dot(df_freq, np.transpose(df_freq)), index=df_freq.index, columns=df_freq.index)

        l = 0
        for pair in it.combinations(df_freq.index, r=2):
           cos = miniresult.at[pair[0], pair[1]] / (miniresult.at[pair[0], pair[0]] * miniresult.at[pair[1], pair[1]]) ** 0.5
           dis_cos_freq.loc[pair[0], pair[1]] = dis_cos_freq.loc[pair[1], pair[0]] = cos
           l += 1
           if l % 100000 == 0:
               print(l // 100000, 'from', 237)


        dis_cos_freq.to_csv(cosinefreq, sep='\t', index=True)
        print(f'cosine matrix saved to file {cosinefreq}')




    def make_tsv_metric(self, cosinefile, savename):
        dis_cos = pd.read_csv(cosinefile, sep='\t', index_col=0)
        df = pd.read_csv('OMIM.gene-all-phenotypes.all.Jan-2024.tsv', sep='\t')
        df = df[df['GeneCounts']>1]
        df['MIM'] = ['OMIM:'+x for x in df['MIM']]
        dis_set = df.groupby('#GeneSymbol').MIM.unique().to_dict()
        inh = df.groupby(['#GeneSymbol', 'MIM']).Inheritance.unique().to_dict()
        name = df.groupby(['#GeneSymbol', 'MIM']).PhenName.unique().to_dict()

        df_D = pd.read_csv('../2023:2024_course_work/data/D_METRIC_WITH_?.tsv', sep='\t')

        phen_cnts = self.phens.groupby(['gene_symbol', 'disease_id']).hpo_id.nunique().to_dict()

        class_df = pd.read_csv('../2023:2024_course_work/MONDO/mondo(current_version 2024) + OMIM 2024 data/CLASS_COS.tsv', sep='\t', index_col=0)
        omimtomondo = pd.read_csv('../2023:2024_course_work/MONDO/mondo(current_version 2024) + OMIM 2024 data/omim_to_mondo.txt', sep='\t').groupby('omim_id').mondo_id.unique().to_dict()



        with open(savename, 'w') as file:
            print('GeneSymbol', 'OMIMID1', 'MONDOID1', 'NAME1', 'INH1', 'PHENCNTS_1', 'OMIMID2', 'MONDOID2', 'NAME2', 'INH2', 'PHENCNTS_2', 'COS', 'STS', 'CLASS_COS',  sep='\t', file=file)
            for key, value in dis_set.items():
                for pair in it.combinations(value, r=2):
                    if (key, pair[0]) in phen_cnts.keys():
                        cnts1 = phen_cnts[(key, pair[0])]
                    else:
                        cnts1 = '?'
                    if (key, pair[1]) in phen_cnts.keys():
                        cnts2 = phen_cnts[(key, pair[1])]
                    else:
                        cnts2 = '?'

                    mondo1 = omimtomondo[pair[0]][0] if pair[0] in omimtomondo.keys() else '?'
                    mondo2 = omimtomondo[pair[1]][0] if pair[1] in omimtomondo.keys() else '?'
                    if mondo1 in class_df.index and mondo2 in class_df.index:
                        class_cos = class_df.at[mondo1, mondo2]
                    else:
                        class_cos = '?'

                    if f'{key},{pair[0]}' in dis_cos.index and f'{key},{pair[1]}' in dis_cos.index:
                        cos = dis_cos.loc[f'{key},{pair[0]}', f'{key},{pair[1]}']
                        sts = [df_D[df_D['Gene']==key][df_D['omim_id1']==pair[0]][df_D['omim_id2']==pair[1]].sts_measure.unique()[0]
                                     if df_D[df_D['Gene']==key][df_D['omim_id1']==pair[0]][df_D['omim_id2']==pair[1]].shape[0]!=0 else
                                     df_D[df_D['Gene']==key][df_D['omim_id1']==pair[1]][df_D['omim_id2']==pair[0]].sts_measure.unique()[0]]

                        print(key, pair[0], mondo1, name[(key, pair[0])][0], inh[(key, pair[0])][0], cnts1, pair[1], mondo2, name[(key, pair[1])][0], inh[(key, pair[1])][0], cnts2, cos, sts[0], class_cos, sep='\t', file=file)

                    elif pair[0] != 'OMIM:?' and pair[1] != 'OMIM:?':
                        cos = '?'
                        sts = [df_D[df_D['Gene'] == key][df_D['omim_id1'] == pair[0]][df_D['omim_id2'] == pair[1]].sts_measure.unique()[0]
                               if df_D[df_D['Gene'] == key][df_D['omim_id1'] == pair[0]][df_D['omim_id2'] == pair[1]].shape[0] != 0 else
                               df_D[df_D['Gene'] == key][df_D['omim_id1'] == pair[1]][df_D['omim_id2'] == pair[0]].sts_measure.unique()[0]]

                        print(key, pair[0], mondo1, name[(key, pair[0])][0], inh[(key, pair[0])][0], cnts1, pair[1], mondo2,
                              name[(key, pair[1])][0], inh[(key, pair[1])][0], cnts2, cos, sts[0], class_cos, sep='\t',
                              file=file)
                    else:
                        print(key, pair[0], mondo1, name[(key, pair[0])][0], inh[(key, pair[0])][0], cnts1, pair[1], mondo2,
                              name[(key, pair[1])][0], inh[(key, pair[1])][0], cnts2, '?', '?', class_cos, sep='\t',
                              file=file)
        file.close()



    def phenseries(self):
        phser = pd.read_csv('phenseries_titiles.tsv', sep='\t').groupby('Phenotypic Series number')['Phenotypic Series Title'].unique().to_dict()
        sername = pd.read_csv('phenseries_titiles.tsv', sep='\t').groupby('Phenotypic Series number')['Phenotypic Series Title'].unique().to_dict()
        for key, value in phser.items():
            value=value[0]
            for symbol in ['(', ')', ',']:
                value = value.replace(symbol, ' ')
            phser[key] = value

        phser = {k: v for k, v in sorted(phser.items(), key=lambda x: -len(x[1]))}


        df = pd.read_csv('OMIM.gene-all-phenotypes.all.Jan-2024.tsv', sep='\t')
        df['MIM'] = ['OMIM:' + x for x in df['MIM']]
        inh = df.groupby(['#GeneSymbol', 'MIM']).Inheritance.unique().to_dict()
        name = df.groupby(['#GeneSymbol', 'MIM']).PhenName.unique().to_dict()
        class_df = pd.read_csv(
            '../2023:2024_course_work/MONDO/mondo(current_version 2024) + OMIM 2024 data/CLASS_COS.tsv', sep='\t',
            index_col=0)
        omimtomondo = pd.read_csv(
            '../2023:2024_course_work/MONDO/mondo(current_version 2024) + OMIM 2024 data/omim_to_mondo.txt',
            sep='\t').groupby('omim_id').mondo_id.unique().to_dict()
        phen_cnts = self.phens.groupby(['gene_symbol', 'disease_id']).hpo_id.nunique().to_dict()


        print('starting phen series')
        res = [False]*df.shape[0]
        phendisname = {}
        for index, row in df.iterrows():
            disname = row['PhenName']
            for symbol in ['(', ')', ',']:
                disname = disname.replace(symbol, ' ')
            for key, value in phser.items():
                if set(disname.split()) & set(value.split()) == set(value.split()) and len(set(disname.split())) <= len(set(value.split()))+3:
                    res[index] = key
                    phendisname[(key, row['MIM'])] = row['PhenName']
                    break
            if not res[index]:
                res[index] = '?'


        df['phenseries'] = res
        df = df[df['phenseries']!='?']

        print('series added')
        print(df)


        series = {k:v for k,v in filter(lambda x: len(x[1])>1, df.groupby('phenseries')[['#GeneSymbol', 'MIM']].apply(lambda g: g.values.tolist()).to_dict().items())}
        for key, value in series.items():
            value = [f'{pair[0]},{pair[1]}' for pair in value]
            series[key] = value

        print('done: series -> Gene,OMIMID')

        names = {'cosine_1_edited.tsv': 'FREQ_NO', 'mean/cosine_freq_edited.tsv': 'FREQ_MEAN', '01/cosine_freq_edited.tsv': 'FREQ_01', '05/cosine_freq_edited.tsv': 'FREQ_05', '1/cosine_freq_edited.tsv': 'FREQ_1'}

        print('starting a metric')
        with open('phenseries_edited.tsv', 'w') as out:
            print('GENE1', 'OMIMID1', 'MONDOID1', 'NAME1', 'INH1', 'GENE2', 'OMIMID2', 'MONDOID2', 'NAME2', 'INH2', 'SERIESID', 'SERIESNAME', 'TYPE', 'COS', 'CLASS_COS', 'PHEN_CNTS1', 'PHEN_CNTS2', sep='\t', file=out)
            for file in ['cosine_1_edited.tsv', 'mean/cosine_freq_edited.tsv', '01/cosine_freq_edited.tsv', '05/cosine_freq_edited.tsv', '1/cosine_freq_edited.tsv']:
            #for file in ['mean/cosine_freq_edited.tsv']:
                cosine = pd.read_csv(file, index_col=0, sep='\t')
                print('phenseries with file', file)
                for key, value in series.items():
                    for pair in it.combinations(value, r=2):
                        if (pair[0].split(',')[0], pair[0].split(',')[1]) in phen_cnts.keys():
                            cnts1 = phen_cnts[(pair[0].split(',')[0], pair[0].split(',')[1])]
                        else:
                            cnts1 = '?'
                        if (pair[1].split(',')[0], pair[1].split(',')[1]) in phen_cnts.keys():
                            cnts2 = phen_cnts[(pair[1].split(',')[0], pair[1].split(',')[1])]
                        else:
                            cnts2 = '?'

                        mondo1 = omimtomondo[pair[0].split(',')[1]][0] if pair[0].split(',')[1] in omimtomondo.keys() else '?'
                        mondo2 = omimtomondo[pair[1].split(',')[1]][0] if pair[1].split(',')[1] in omimtomondo.keys() else '?'

                        if mondo1 in class_df.index and mondo2 in class_df.index:
                            class_cos = class_df.at[mondo1, mondo2]
                        else:
                            class_cos = '?'

                        if pair[0] in cosine.index and pair[1] in cosine.index:
                            cos = cosine.at[pair[0], pair[1]]
                        else:
                            cos = '?'

                        print(pair[0].split(',')[0], pair[0].split(',')[1], mondo1, phendisname[(key, pair[0].split(',')[1])],
                          inh[(pair[0].split(',')[0], pair[0].split(',')[1])][0], pair[1].split(',')[0], pair[1].split(',')[1], mondo2, phendisname[(key, pair[1].split(',')[1])],
                          inh[(pair[1].split(',')[0], pair[1].split(',')[1])][0], key, sername[key][0], names[file], cos, class_cos, cnts1, cnts2, sep='\t', file=out)


    def randomsample(self):
        df = pd.read_csv('OMIM.gene-all-phenotypes.all.Jan-2024.tsv', sep='\t')
        df['MIM'] = ['OMIM:' + x for x in df['MIM']]
        inh = df.groupby(['#GeneSymbol', 'MIM']).Inheritance.unique().to_dict()
        name = df.groupby(['#GeneSymbol', 'MIM']).PhenName.unique().to_dict()
        class_df = pd.read_csv(
            '../2023:2024_course_work/MONDO/mondo(current_version 2024) + OMIM 2024 data/CLASS_COS.tsv', sep='\t',
            index_col=0)
        omimtomondo = pd.read_csv(
            '../2023:2024_course_work/MONDO/mondo(current_version 2024) + OMIM 2024 data/omim_to_mondo.txt',
            sep='\t').groupby('omim_id').mondo_id.unique().to_dict()
        phen_cnts = self.phens.groupby(['gene_symbol', 'disease_id']).hpo_id.nunique().to_dict()
        dis_set = df.groupby('#GeneSymbol').MIM.unique().to_dict()

        alls = []
        for key, value in dis_set.items():
            for x in value:
                alls.append(f'{key},{x}')


        randdif = [x for x in filter(lambda x: x[0].split(',')[0] != x[1].split(',')[0], [x for x in it.combinations(alls, r=2)])]
        names = {'cosine_1_edited.tsv': 'FREQ_NO', 'mean/cosine_freq_edited.tsv': 'FREQ_MEAN', '01/cosine_freq_edited.tsv': 'FREQ_01',
                     '05/cosine_freq_edited.tsv': 'FREQ_05', '1/cosine_freq_edited.tsv': 'FREQ_1'}

        print('starting a metric')
        with open('random_pairs_diff_genes_edited.tsv', 'w') as out:
            print('GENE1', 'OMIMID1', 'MONDOID1', 'NAME1', 'INH1', 'GENE2', 'OMIMID2', 'MONDOID2', 'NAME2', 'INH2',
                      'TYPE', 'COS', 'CLASS_COS', 'PHEN_CNTS1', 'PHEN_CNTS2', sep='\t',
                      file=out)
            #for file in ['mean/cosine_freq_edited.tsv']:
            for file in ['cosine_1_edited.tsv', 'mean/cosine_freq_edited.tsv', '01/cosine_freq_edited.tsv', '05/cosine_freq_edited.tsv', '1/cosine_freq_edited.tsv']:
                cosine = pd.read_csv(file, index_col=0, sep='\t')
                print('different genes with file', file)
                for pair in random.sample(randdif, 5000):
                    if (pair[0].split(',')[0], pair[0].split(',')[1]) in phen_cnts.keys():
                                cnts1 = phen_cnts[(pair[0].split(',')[0], pair[0].split(',')[1])]
                    else:
                                cnts1 = '?'
                    if (pair[1].split(',')[0], pair[1].split(',')[1]) in phen_cnts.keys():
                                cnts2 = phen_cnts[(pair[1].split(',')[0], pair[1].split(',')[1])]
                    else:
                                cnts2 = '?'

                    mondo1 = omimtomondo[pair[0].split(',')[1]][0] if pair[0].split(',')[
                                                                                  1] in omimtomondo.keys() else '?'
                    mondo2 = omimtomondo[pair[1].split(',')[1]][0] if pair[1].split(',')[
                                                                                  1] in omimtomondo.keys() else '?'

                    if mondo1 in class_df.index and mondo2 in class_df.index:
                                class_cos = class_df.at[mondo1, mondo2]
                    else:
                                class_cos = '?'

                    if pair[0] in cosine.index and pair[1] in cosine.index:
                                cos = cosine.at[pair[0], pair[1]]
                    else:
                                cos = '?'

                    print(pair[0].split(',')[0], pair[0].split(',')[1], mondo1,
                                  name[(pair[0].split(',')[0], pair[0].split(',')[1])][0],
                                  inh[(pair[0].split(',')[0], pair[0].split(',')[1])][0], pair[1].split(',')[0],
                                  pair[1].split(',')[1], mondo2, name[(pair[1].split(',')[0], pair[1].split(',')[1])][0],
                                  inh[(pair[1].split(',')[0], pair[1].split(',')[1])][0],
                                  names[file], cos, class_cos, cnts1, cnts2, sep='\t', file=out)





# make all the metrics!
hpo = HPO('hp-base.is_a.tsv', '../2023:2024_course_work/HPO/hpo_cos(current version 2024)/OMIM_genes_to_phenotype.txt')
# for mean filling
hpo.frequency()
hpo.OMIMxHPO_freq('mean/cosine_freq_edited.tsv')
hpo.make_tsv_metric('mean/cosine_freq_edited.tsv', 'mean/metric_freq_edited.tsv')

# no frequency used in metric
hpo.OMIMxHPO_1('cosine_1_edited.tsv')
hpo.make_tsv_metric('cosine_1_edited.tsv', 'metric_1_edited.tsv')

# make random and phnotypic series data for validation
hpo.randomsample()
hpo.phenseries()


# fill with 0.1
hpo = HPO('hp-base.is_a.tsv', '../2023:2024_course_work/HPO/hpo_cos(current version 2024)/OMIM_genes_to_phenotype.txt')
hpo.frequency(0.1)
hpo.OMIMxHPO_freq('01/cosine_freq_edited.tsv')
hpo.make_tsv_metric('01/cosine_freq_edited.tsv', '01/metric_freq_edited.tsv')

# fill with 0.5
hpo = HPO('hp-base.is_a.tsv', '../2023:2024_course_work/HPO/hpo_cos(current version 2024)/OMIM_genes_to_phenotype.txt')
hpo.frequency(0.5)
hpo.OMIMxHPO_freq('05/cosine_freq_edited.tsv')
hpo.make_tsv_metric('05/cosine_freq_edited.tsv', '05/metric_freq_edited.tsv')

# fill with 1
hpo = HPO('hp-base.is_a.tsv', '../2023:2024_course_work/HPO/hpo_cos(current version 2024)/OMIM_genes_to_phenotype.txt')
hpo.frequency(1.0)
hpo.OMIMxHPO_freq('1/cosine_freq_edited.tsv')
hpo.make_tsv_metric('1/cosine_freq_edited.tsv', '1/metric_freq_edited.tsv')





