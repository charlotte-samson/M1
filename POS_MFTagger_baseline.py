from collections import defaultdict
import operator

def train():
    word2Pos2count = defaultdict(lambda: defaultdict(int))
    with open("fr-ud-train.conllu", "r") as tf:
        for line in tf.readlines():
            line_content = line.split('\t')
            if line_content[0].isdigit():
                # Skipping 'aux' type lines
                word2Pos2count[line_content[1]][line_content[3]]+=1
    return word2Pos2count

def test(word2Pos, filepath):
    good_answers = 0
    with open(filepath, "r") as tf:
        lines = tf.readlines()
        for line in lines:
            line_content = line.split('\t')
            if line_content[0].isdigit():
                if word2Pos[line_content[1]]==line_content[3]:
                    good_answers +=1
                elif word2Pos[line_content[1]]=="":
                    if word2Pos["chat"]==line_content[3]:
                        good_answers+=1
        print(good_answers/len(lines)*100)

word2Pos2Count = train()

word2Pos = defaultdict(str)

for kei in word2Pos2Count.keys():
    word2Pos[kei] = max(word2Pos2Count[kei].items(), key=operator.itemgetter(1))[0]

test(word2Pos, "fr-ud-train.conllu")
test(word2Pos, "fr-ud-dev.conllu")
test(word2Pos, "fr-ud-test.conllu")