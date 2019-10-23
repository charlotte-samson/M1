import numpy as np


class Data:
    def __init__(self, word2vec, train, dev, test):
        self.train_file = train
        self.dev_file = dev
        self.test_file = test
        self.Y_train, self.X_train = 0, 0
        self.Y_dev, self.X_dev = 0, 0
        self.Y_test, self.X_test = 0, 0
        self.word2vec = word2vec
        self.pos_semantics = ["NONE"]
        self.extract_train()
        self.extract_dev()
        self.extract_test()
        self.trainwords = []
        self.devwords = []
        self.testwords = []

    def extract_train(self):
        with open(self.train_file, 'r') as tf:
            for line in tf.readlines():
                line_content = line.split('\t')
                if line_content[0].isdigit():
                    # Skipping 'aux' type lines
                    if '-' in line_content[0]:
                        continue
                    #Extracting tags from conllu
                    if line_content[3] not in self.pos_semantics:
                        self.pos_semantics.append(line_content[3])
        self.X_train, self.Y_train, self.trainwords = self.extract(self.train_file, self.X_train, self.Y_train)

    def extract_dev(self):
        self.X_dev, self.Y_dev, self.devwords = self.extract(self.dev_file, self.X_dev, self.Y_dev)

    def extract_test(self):
        self.X_test, self.Y_test, self.testwords = self.extract(self.test_file, self.X_test, self.Y_test)

    def extract(self, file_conll, X_corp, Y_corp):
        corpwords = []
        with open(file_conll, 'r') as cf:
            lines = cf.readlines()
            #initializing matrixes
            X_corp = np.zeros((len(lines), 50))
            Y_corp = np.zeros((len(lines), len(self.pos_semantics)))
            rows_to_delete = [0]
            for i in range(len(lines)):
                line_content = lines[i].split('\t')
                if line_content[0].isdigit():
                    # Skipping 'aux' type lines
                    if '-' in line_content[0]:
                        rows_to_delete.append(i)
                        continue
                    #Extracting words from the corpus
                    corpwords.append(line_content[1])
                    if len(self.word2vec[line_content[2]]) == 0:
                        X_corp[i] = self.word2vec["chat"]
                    else:
                    #filling the matrix with the embeddings
                        X_corp[i] = np.array(self.word2vec[line_content[2]])
                    #filling the tag matrix
                    pos_vec_to_add = [0] * (len(self.pos_semantics))
                    pos_vec_to_add[self.pos_semantics.index(line_content[3])] = 1
                    Y_corp[i] = pos_vec_to_add
                else:
                    rows_to_delete.append(i)
        #Deleting useless lines
        X_corp = np.delete(X_corp, rows_to_delete, axis=0)
        Y_corp = np.delete(Y_corp, rows_to_delete, axis=0)
        return X_corp, Y_corp, corpwords

