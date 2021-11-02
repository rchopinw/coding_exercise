from collections import Counter, defaultdict


def find_anagram(words, target):
    result, count = [], Counter(target)
    for word in words:
        if Counter(word) == count:
            result.append(word)
    return result


class AnagramFinder:
    def __init__(self, words):
        self.words = words
        self.ws_map = self.__formulate_collections(self.words)

    def find(self, s):
        s_rep = self.__gen_represent(s)
        return self.ws_map.get(s_rep)

    def __formulate_collections(self, ws):
        d = defaultdict(list)
        for w in ws:
            d[self.__gen_represent(w)].append(w)
        return d

    def __gen_represent(self, w):
        represent = [0 for _ in range(26)]
        for c in w:
            represent[ord(c) - ord('a')] += 1
        return tuple(represent)

