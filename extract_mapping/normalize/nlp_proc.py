'''
** Need to edit to make "get_categories" method occur before mapping (should separate income/expense before the mapping)
'''

import pandas as pd
import numpy as np
import spacy
import organize

# Initialize spaCY
#nlp = spacy.load("en_core_web_trf")
nlp = spacy.load("en_core_web_lg")

# Create similarity matrix
def get_sim_matrix(knowledge_base, unmapped_phrases):
    # Create pipeline
    kb_pipe = list(nlp.pipe(knowledge_base))
    to_map_pipe = list(nlp.pipe(unmapped_phrases))
    
    # Compute similarity
    sim_mat = [[x.similarity(y) for x in to_map_pipe] for y in kb_pipe]
    sim_mat = pd.DataFrame(sim_mat, index=knowledge_base, columns=unmapped_phrases)
    
    return sim_mat

# Get max similarity value and item
def get_max_sim(similarity_matrix):
    max_sim = similarity_matrix.max()
    kb_match = similarity_matrix.idxmax()
    max_sim = pd.concat([max_sim, kb_match], axis=1).rename(columns={0:'a',1:'b'}).reset_index()
    max_sim.columns = ['unmapped_items','max_sim_val', 'kb_match']
    
    return max_sim

# Modify to be clear that's Op Statement Specific
# Function to discover most likely column feature opex items
# Infer that column with max sum of similarity to CREFC is the items list
def items_mapper(opex_df, knowledge_base):
    # for each column in df, if not numeric, run similarity matrix
    max_sum_sim = 0
    for i in range(len(opex_df.columns)):
        # Check if column is string
        if opex_df.iloc[:,i].dtype == object:
            # Compute sum of max similarity
            sim_mat = get_sim_matrix(knowledge_base, opex_df.iloc[:,i].dropna())
            max_sim = get_max_sim(sim_mat)
            sum_sim = max_sim.max_sim_val.sum()

            if sum_sim > max_sum_sim:
                max_sim_final = max_sim
                max_sum_sim = sum_sim
                # Set column as containing opex index
                items_col_idx = i
    # Return index of column containing opex items and crefc mapping
    return items_col_idx, max_sim_final

# Modify to be clear that's Op Statement Specific
# Organize table with main and subcategories
def get_categories(structured_table):
    # Create subset containing all NA
    # drop if index diff -1 = 1 delete (want to keep header directly above numerical values)
    categories = structured_table.iloc[:,1:].copy()
    categories = categories[categories.isnull().all(axis=1)].reset_index(drop=False)

    # Compute forward and backward sequence until next header utilizing index
    categories['index_seq_diff'] = categories['index'].diff()
    categories['index_seq_diff_fwd'] = categories['index'].diff(periods=-1)*-1

    # Select headers to keep (the one nearest to next set of data)
    categories = categories[(categories.index_seq_diff>=1) & ((categories.index_seq_diff_fwd > 1) | (categories.index_seq_diff_fwd.isnull()==True))]

    # Assign op statement category/group according to header indexes
    categories = categories.merge(structured_table.iloc[:,0], left_on='index', right_index=True)
    categories = categories[['index',categories.columns[-1]]]
    categories.columns = ['index','category']
    categories.set_index('index', inplace=True)
    
    main = structured_table.merge(categories, left_index=True, right_index=True, how='left')

    # Tag master categories as those with all NA except column 0
    main['master_category'] = np.where(main.iloc[:,1:].isnull().all(axis=1), main.iloc[:,0], np.NaN)

    # Forward fill categories
    main.category = main.category.fillna(method='ffill')
    main.master_category = main.master_category.fillna(method='ffill')
    
    # Create dummy column for whether an item is a total or not
    main['total'] = np.where(main.iloc[:,0].str.contains('total', case=False), 1, 0)
    
    # Reorder columns so can remove unneeded cells systematically
    cols = main.columns.to_list()
    cols = [cols[0]] + cols[-3:] + cols[1:-3]
    main = main[cols]
    '''    if len(cols[1:-3]) > 1:
        print("1")
        cols = [cols[0]] + cols[-3:] + cols[1:-3]
    else:
        print("2")
        cols = [cols[0]] + cols[-3:] + [cols[1:-3]]
    try:    
        main = main[cols]
    except:
        pass
    '''
    
    # Drop rows where 4: are ALL NaN (i.e. rows just with category labels etc.)
    main = main.dropna(subset=main.iloc[:,4:].columns, how='all').reset_index(drop=True)
    
    # Rename opex items column
    main.rename(columns={main.columns[0]: 'opex_items'}, inplace=True)
    
    return main

    
class header_detector:
    def __init__(self, table, knowledge_base):
        self.table = table
        self.knowledge_base = knowledge_base

    def get_max_sim(self):
        for i in range(len(self.table.columns)):
            # Compute simmat and get max similarity for each column
            sim_mat = get_sim_matrix(self.knowledge_base, self.table.iloc[:,i].astype('str'))
            max_sim = get_max_sim(sim_mat)

            # Keep max_sim_val, row index and col index
            if i == 0:
                df_sim = max_sim[['max_sim_val']].rename(columns={'max_sim_val':'0'})
                #df_sim.columns = ['row_idx','0']
                #max_sim.rename(columns={'index':'row_idx'}, inplace=True)
            else:
                df_sim[[i]] = max_sim.max_sim_val
        return df_sim

    def get_header_candidates(self):
        '''
        V1 uses a z-score cut-off of +1 to label header... training to be conducted to tune this hyperparameter
        '''
        # Get sum across each row, compute zscores
        # min/max of first continuous set of indexes as header start/end
        df_sim = self.get_max_sim()
        col_sum = df_sim.sum(axis=1)
        mean = np.mean(col_sum)
        std = np.std(col_sum)
        zscores = []
        zscores_full = []
        # i = row index; v = max similarity score ;z = zscore
        for i,v in enumerate(col_sum):
            z = (v-mean)/std
            zscores_full.append([i,v,z])
            if np.abs(z) > 1:
                zscores.append([i,v,z])
        return zscores, zscores_full

    def get_num_data_start(self):
        # Coerce to numeric
        num_table = self.table.apply(lambda x: organize.numeric_cleaner(x))
        # Cut rows of num_table that are all NaN
        num_table = num_table.dropna(how='all')
        # Get index where numeric data begins
        data_start = num_table.index[0]
        return data_start

    # Modify this one to capture zscore >1 and diff sequentially less than 0.5 or so.
    def get_header_start_end(self):
        # Calculate the difference in zscores for each row 
        _,zscores_full =  self.get_header_candidates()
        z_df = pd.DataFrame(zscores_full, columns=['idx','v','z'])
        z_df['deltaz'] = z_df.z.diff()

        # Keep rows until the jump in zscore diff is significant
        z_df['deltaz_jump'] = np.where(np.abs(z_df.deltaz) > 0.2, 1, 0)
        end = z_df[z_df.deltaz_jump==1].idx.min()-1
        start = 0

        # Ensure header end < data_start
        data_start = self.get_num_data_start()
        if end < data_start:
            pass
        else:
            end = data_start-1
        # TODO: contemplate if always starting header at 0 is ok - this version assumes rows with all but 1 NA are deleted.
        return [start, end]


    def replace_header(self):
        # Clean header
        start_end = self.get_header_start_end()
        header = self.table.iloc[start_end[0]:start_end[1]+1,]
        header = header.apply(lambda x: x.str.strip())

        # Replace header with detected header and drop from data
        clean_df = self.table.copy()
        for c in clean_df.columns:
            clean_df = clean_df.rename(columns={c: header[c].str.cat(sep=' ')})
        clean_df = clean_df.drop(start_end).reset_index(drop=True)
        return clean_df

    # Original methodology just grabbed sequential rows with zscore>1 OBSOLETE
    def get_header_start_end_OLD(self):
        # Isolate min/max of first continuous set of indexes as header start/end
        zscores,_ =  self.get_header_candidates()
        # get row index
        idx_list = [elem[0] for elem in zscores]
        # find start and end of continuous set
        # Faster method may be to compute delta of everything, stop when >1 or something involving min/max
        for i,v in enumerate(idx_list):
            if i == 0:
                start = v
                end = v
            else:
                delta = v - end
                if delta <= 1:
                    end = v
                else:
                    break
        # TODO: modify code to return max_sim and zscore too
        return [start, end]