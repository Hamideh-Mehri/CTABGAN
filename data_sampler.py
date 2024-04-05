import numpy as np


def random_choice_prob_index_sampling(probs,col_idx):
    
    """
    Used to sample a specific category within a chosen one-hot-encoding representation 
    Inputs:
    1) probs -> probability mass distribution of categories 
    2) col_idx -> index used to identify any given one-hot-encoding
    
    Outputs:
    1) option_list -> list of chosen categories 
    
    """

    option_list = []
    for i in col_idx:
        # for improved stability
        pp = probs[i] + 1e-6 
        pp = pp / sum(pp)
        # sampled based on given probability mass distribution of categories within the given one-hot-encoding 
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""
    
    #output_info is transformer.output_info_list, data is transformed train data
    def __init__(self, data, output_info):
   
        self.model = []
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0 
        self.p_log_sampling = []  
        self.p_sampling = [] 
        self.output_info_flat = [elem for sublist in output_info for elem in sublist]
        self._data = data
        st = 0
        for column_info in self.output_info_flat:
            # ignoring columns that do not represent one-hot-encodings
            if column_info.activation_fn == 'tanh':
               st += column_info.dim
               continue
            elif column_info.activation_fn == 'softmax':
                # using starting (st) and ending (ed) position of any given one-hot-encoded representation to obtain relevant information
                ed = st + column_info.dim
                self.model.append(np.argmax(data[:, st:ed], axis=-1)) #list containing an index of highlighted categories in their corresponding one-hot-encoded represenations
                self.interval.append((self.n_opt,column_info.dim))
                self.n_col += 1
                self.n_opt += column_info.dim
                freq = np.sum(data[:, st:ed], axis=0)  
                log_freq = np.log(freq + 1)  
                log_pmf = log_freq / np.sum(log_freq)
                self.p_log_sampling.append(log_pmf)
                pmf = freq / np.sum(freq)
                self.p_sampling.append(pmf)
                st = ed
        # Compute _rid_by_cat_cols
        # data -> real transformed input data
        # _rid_by_cat_cols is a list of lists - >if we have 15 columns with one-hot code representation, we have 15 lists in _rid_by_cat_cols, each of these 15 lists are list of lists.
        # -> the number of these lists is equal to the length of one-hot encoded representation of that column. , the length of these lists are different, for example, the 
        # i'th list contains the row index which the i'th element of the one-hot encoding representation is equal to 1. 
        
        self._rid_by_cat_cols = []
        st = 0
        for column_info in self.output_info_flat:
            if column_info.activation_fn == 'tanh':
                st += column_info.dim
                continue
            elif column_info.activation_fn == 'softmax':
                ed = st + column_info.dim
                rid_by_cat = []
                for j in range(column_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
        
    
    def sample_condvec_train(self, batch):
        
        """
        Used to create the conditional vectors for feeding it to the generator during training
        Inputs:
        1) batch -> no. of data records to be generated in a batch
        Outputs:
        1) vec -> a matrix containing a conditional vector for each data point to be generated 
        2) mask -> a matrix to identify chosen one-hot-encodings across the batch
        3) idx -> list of chosen one-hot encoding across the batch
        4) opt1prime -> selected categories within chosen one-hot-encodings
        """

        if self.n_col == 0:
            return None
        batch = batch
        
        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')

        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # matrix of shape (batch x total no. of one-hot-encoded representations) with 1 in indexes of chosen representations and 0 elsewhere
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        
        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_log_sampling,idx) 
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):
            vec[i, self.interval[idx[i]][0] + opt1prime[i]] = 1
            
        return vec, mask, idx, opt1prime

    def sample_condvec(self, batch):
        
        """
        Used to create the conditional vectors for feeding it to the generator after training is finished
        Inputs:
        1) batch -> no. of data records to be generated in a batch
        Outputs:
        1) vec -> an array containing a conditional vector for each data point to be generated 
        """

        if self.n_col == 0:
            return None
        
        batch = batch

        # each conditional vector in vec is a one-hot vector used to highlight a specific category across all possible one-hot-encoded representations 
        # (i.e., including modes of continuous and mixed columns)
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        
        # choosing one specific one-hot-encoding from all possible one-hot-encoded representations 
        idx = np.random.choice(np.arange(self.n_col), batch)

        # producing a list of selected categories within each of selected one-hot-encoding representation
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        # assigning the appropriately chosen category for each corresponding conditional vector
        for i in np.arange(batch):   
            vec[i, self.interval[idx[i]][0] + opt1prime[i]] = 1
            
        return vec

    def sample_data(self, n, col, opt):
        # col is list of column indexes and opt is the id of selected categories in that columns
        
        # sample the transformed real data according to the conditional vector 
      
        
        """Sample data from original training data satisfying the sampled conditional vector.
        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return self._data[idx]


                