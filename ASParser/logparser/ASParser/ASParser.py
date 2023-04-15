import re
import pandas as pd
import os
import math
from timeit import default_timer as timer
from logparser.ASParser.sim import sim_log_node, find_brackets, sim_log2log, sim_node_node, SplitFirstLayer, fold_nodes, \
    can_fold
import scipy.special


class Node:
    def __init__(self, logIDL, word, delimiters, brackets, cid=-1):
        self.logIDL = logIDL.copy()
        self.values = set()
        self.pattern = ""
        self.word = word
        self.children = []
        self.tooManyVals = False
        self.isLeaf = False
        self.delimiters = delimiters
        self.brackets = brackets
        self.delimiter_now = None
        self.is16 = False
        self.hasDigit = False
        self.score = -1.0
        self.firstWord = word
        self.becomeWildcard = False
        self.String = ""
        self.cid = cid
        self.parent = None
        self.removed = False

        self.inWildcardTalble = False
        self.last = ""

        if (self.word == '-<\\d>'):
            self.word = '<\\d>'
        if (not delimiters and not brackets):
            self.makeLeaf()
            if re.match(r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b',
                        word) or word == '<\d>' or word == '-<\d>':
                self.is16 = True
            if (re.search(r'\d', word)):
                self.hasDigit = True

        self.values.add(self.word)
        self.value_logIDL = {self.word: cid}

    def foldChildren(self):
        self.children = []
        self.pattern = ""

    def copy(self):
        newNode = Node(logIDL=self.logIDL, word=self.word, delimiters=self.delimiters, brackets=self.brackets,
                       cid=self.cid)
        newNode.logIDL = self.logIDL
        newNode.isLeaf = self.isLeaf
        newNode.tooManyVals = self.tooManyVals
        newNode.pattern = self.pattern
        newNode.values = self.values
        newNode.value_logIDL = self.value_logIDL.copy()
        newNode.delimiter_now = self.delimiter_now
        newNode.is16 = self.is16
        newNode.hasDigit = self.hasDigit
        newNode.word = self.firstWord
        newNode.becomeWildcard = self.becomeWildcard
        newNode.String = self.String
        newNode.cid = self.cid
        newNode.parent = self.parent
        newNode.removed = self.removed

        if len(self.children) == 0:
            return newNode
        for child in self.children:
            newNode.children.append(child.copy())
        return newNode

    def expandByDelimiter(self, delimiter_use):
        self.delimiter_now = delimiter_use
        if (delimiter_use == "#brackets#"):
            pos = find_brackets(self.brackets)
            if (not pos):
                return
            if (pos[0] != 0):
                pos.insert(0, -1)
            if (pos[-1] != len(self.word) - 1):
                pos.append(len(self.word))
            log = self.word
            self.pattern = ""
            last_pos = -1
            for i in range(len(pos)):
                if (pos[i] == last_pos + 1):
                    self.pattern += log[pos[i]]
                else:
                    self.pattern += "F"
                    self.pattern += log[pos[i]]
                last_pos = pos[i]
            if (pos[-1] != len(log) - 1):
                self.pattern += "F"

        else:
            if (delimiter_use not in self.delimiters.keys()):
                return
            pos = self.delimiters[delimiter_use].copy()
            if (self.brackets):
                symbols_remove = []
                for brack in self.brackets.keys():
                    list_ = self.brackets[brack]
                    for tuple2 in list_:
                        for s in pos:
                            if (s > tuple2[0] and s < tuple2[1]):
                                symbols_remove.append(s)
                symbols_remove = set(symbols_remove)
                for index in symbols_remove:
                    pos.remove(index)
            if (not pos):
                return
            if (pos[0] != 0):
                pos.insert(0, -1)
            if (pos[-1] != len(self.word) - 1):
                pos.append(len(self.word))
            self.pattern = ""
            if (pos[0] == 0):
                self.pattern += delimiter_use
            for i in range(len(pos) - 2):
                self.pattern += "F"
                self.pattern += delimiter_use
            self.pattern += "F"
            if (pos[-1] == len(self.word) - 1):
                self.pattern += delimiter_use

        for i in range(len(pos) - 1):
            log_new = self.word[(pos[i] + 1):pos[i + 1]]
            sub_delimiters = {}
            for key in self.delimiters:
                list_new = []
                list_old = self.delimiters[key]
                for posa in list_old:
                    if (posa >= pos[i] + 1 and posa < pos[i + 1]):
                        list_new.append(posa - (pos[i] + 1))
                if len(list_new) > 0:
                    sub_delimiters[key] = list_new
            sub_bracket = {}
            for key in self.brackets:
                list_new = []
                list_old = self.brackets[key]
                for tup in list_old:
                    if (tup[0] >= pos[i] + 1 and tup[1] < pos[i + 1]):
                        list_new.append([tup[0] - (pos[i] + 1), tup[1] - (pos[i] + 1)])
                if len(list_new) > 0:
                    sub_bracket[key] = list_new
            newNode = Node(self.logIDL, log_new, sub_delimiters, sub_bracket, cid=self.cid)
            self.children.append(newNode)
            newNode.parent = self

    def toString(self, templateTree):
        if self.isLeaf or len(self.children) == 0:
            if len(self.values) == 1:
                value = list(self.values)[0]
            else:
                value = "<*>"
                values = list(self.values)
                templateTree.wildcards_node[templateTree.wid] = self
                templateTree.wildcards_values[templateTree.wid] = values
                templateTree.wid += 1
            self.String = value
            return value
        else:
            nodeStr = ""
            child_id = 0
            for ch in self.pattern:
                if ch != "F":
                    nodeStr += ch
                else:
                    value = self.children[child_id].toString(templateTree)
                    nodeStr += value
                    child_id += 1
            self.String = nodeStr
            return nodeStr

    def makeLeaf(self):
        self.isLeaf = True
        self.pattern = "F"
        for child in self.children:
            child.makeLeaf()
            child.removed = True
        self.children = []

    def isWildcard(self, value):
        if (re.search(r'\d+', value)):
            return True
        if (is_Date(value)):
            return True
        if (value == '<\\d>'):
            return False
        if (not re.search(r'[a-zA-Z0-9]', value)):
            return False
        if (self.is16):
            return True
        if len(value) < 2:
            return True
        if len(re.findall('[^a-zA-Z0-9._ ]', value)) != 0:
            return True
        return False

    def addValue(self, value, _, cid):
        if self.tooManyVals:
            return
        self.values.add(value)
        if (len(self.values) > 1):
            if (self.word != "<*>"):
                self.word = "<*>"
                self.becomeWildcard = True
            self.last = value

            if (self.becomeWildcard):
                for val in self.values:
                    if (self.isWildcard(val)):
                        self.tooManyVals = True
                        self.value_logIDL = {}
                        self.values = set()
                        return
            if (self.isWildcard(value)):
                self.tooManyVals = True
                self.value_logIDL = {}
                self.values = set()
                return

            if (len(self.values) > 5):
                self.tooManyVals = True
                self.value_logIDL = {}
                self.values = set()
        return

    def canSplit(self):
        if self.tooManyVals or len(self.values) <= 1:
            return False
        if self.word == "<->":
            return False
        if len(self.values) > 5:
            return False
        values = list(self.values)
        has_digit = 0
        for value in values:
            if (re.search(r'\d+', value) or re.search(r'<\\d>', value) or re.fullmatch(r'[a-fA-F]+', value)):
                has_digit += 1
            if (value == '<\\d>'):
                continue
            if (not re.search(r'[a-zA-Z0-9]', value)):
                continue
            if (is_Date(value)):
                return False
            if len(value) < 2:
                return False
            if len(re.findall('[^a-zA-Z0-9._ ]', value)) != 0:
                return False
        if (has_digit == len(values)):
            return False
        return True

    def merge_log(self, logID, log, delimiters, brackets):
        if (log == '-<\\d>'):
            log = '<\\d>'
        if self.isLeaf:
            self.addValue(log, logID, self.cid)
            return
        if log == self.word and len(self.children) == 0:
            self.addValue(log, logID, self.cid)
            return
        if (re.search(r'\d', log) or log == '<\d>'):
            self.hasDigit = True
        if (self.is16):
            if (not (re.match(r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b',
                              log)) or log == '<\d>' or log == '-<\d>'):
                self.is16 = False

        if (self.delimiter_now != None):
            if (self.delimiter_now == "#brackets#"):
                pos = find_brackets(brackets)
                if (not pos):
                    self.makeLeaf()
                    self.tooManyVals = True
                    self.values = set()
                    self.word = "<*>"
                    self.becomeWildcard = False
                    # self.addValue(log, logID,self.cid)
                    return
                if (pos[0] != 0):
                    pos.insert(0, -1)
                if (pos[-1] != len(log) - 1):
                    pos.append(len(log))
            else:
                if (self.delimiter_now not in delimiters.keys()):
                    self.makeLeaf()
                    self.tooManyVals = True
                    self.values = set()
                    self.word = "<*>"
                    self.becomeWildcard = False
                    # self.addValue(log, logID,self.cid)
                    return
                pos = delimiters[self.delimiter_now].copy()
                if (brackets):
                    symbols_remove = []
                    for brack in brackets.keys():
                        list_ = brackets[brack]
                        for tuple2 in list_:
                            for s in pos:
                                if (s > tuple2[0] and s < tuple2[1]):
                                    symbols_remove.append(s)
                    symbols_remove = set(symbols_remove)
                    for index in symbols_remove:
                        pos.remove(index)
                if (not pos):
                    self.makeLeaf()
                    self.tooManyVals = True
                    self.values = set()
                    self.word = "<*>"
                    self.becomeWildcard = False
                    # self.addValue(log, logID,self.cid)
                    return
                if (pos[0] != 0):
                    pos.insert(0, -1)
                if (pos[-1] != len(log) - 1):
                    pos.append(len(log))
            if (len(pos) - 1 != len(self.children)):
                self.makeLeaf()
                self.tooManyVals = True
                self.values = set()
                self.word = "<*>"
                self.becomeWildcard = False
                # self.addValue(log, logID,self.cid)
                return
            else:
                for i in range(len(self.children)):
                    log_new = log[(pos[i] + 1):pos[i + 1]]
                    sub_delimiters = {}
                    for key in delimiters:
                        list_new = []
                        list_old = delimiters[key]
                        for posa in list_old:
                            if (posa >= pos[i] + 1 and posa < pos[i + 1]):
                                list_new.append(posa - (pos[i] + 1))
                        if len(list_new) > 0:
                            sub_delimiters[key] = list_new
                    sub_bracket = {}
                    for key in brackets:
                        list_new = []
                        list_old = brackets[key]
                        for tup in list_old:
                            if (tup[0] >= pos[i] + 1 and tup[1] < pos[i + 1]):
                                list_new.append([tup[0] - (pos[i] + 1), tup[1] - (pos[i] + 1)])
                        if len(list_new) > 0:
                            sub_bracket[key] = list_new
                    self.children[i].merge_log(logID, log_new, sub_delimiters, sub_bracket)
            self.addValue(log, logID, self.cid)
            return
        else:
            log1 = log
            log2 = self.word
            delimiters1 = delimiters
            delimiters2 = self.delimiters
            brackets1 = brackets
            brackets2 = self.brackets

            delimiters_find = [v for v in delimiters1 if v in delimiters2]
            delimiters_same_pattern = []
            pos1 = {}
            pos2 = {}
            for symbol in delimiters_find:
                symbols1 = delimiters1[symbol].copy()
                symbols2 = delimiters2[symbol].copy()
                symbols1_remove = []
                symbols2_remove = []
                for brack in brackets1.keys():
                    list_ = brackets1[brack]
                    for tuple2 in list_:
                        for s in symbols1:
                            if (s > tuple2[0] and s < tuple2[1]):
                                symbols1_remove.append(s)

                for brack in brackets2.keys():
                    list_ = brackets2[brack]
                    for tuple2 in list_:
                        for s in symbols2:
                            if (s > tuple2[0] and s < tuple2[1]):
                                symbols2_remove.append(s)
                symbols1_remove = set(symbols1_remove)
                symbols2_remove = set(symbols2_remove)

                for index in symbols1_remove:
                    symbols1.remove(index)
                for index in symbols2_remove:
                    symbols2.remove(index)
                if (len(symbols1) == len(symbols2) and len(symbols1) != 0 and len(symbols2) != 0):
                    delimiters_same_pattern.append(symbol)
                    pos1[symbol] = symbols1
                    pos2[symbol] = symbols2
            if (brackets1 and brackets2 and not delimiters_same_pattern):
                split_pos1 = find_brackets(brackets1)
                split_pos2 = find_brackets(brackets2)
                self.pattern = ""
                last_pos = -1
                for i in range(len(split_pos2)):
                    if (split_pos2[i] == last_pos + 1):
                        self.pattern += log2[split_pos2[i]]
                    else:
                        self.pattern += "F"
                        self.pattern += log2[split_pos2[i]]
                    last_pos = split_pos2[i]
                if (split_pos2[-1] != len(log2) - 1):
                    self.pattern += "F"

                if (split_pos1[0] != 0):
                    split_pos1.insert(0, -1)
                if (split_pos1[-1] != len(log1) - 1):
                    split_pos1.append(len(log1))
                if (split_pos2[0] != 0):
                    split_pos2.insert(0, -1)
                if (split_pos2[-1] != len(log2) - 1):
                    split_pos2.append(len(log2))
                if (len(split_pos1) != len(split_pos2)):
                    self.makeLeaf()
                    self.addValue(log, logID, self.cid)
                    return
                self.delimiter_now = '#brackets#'
                for i in range(len(split_pos1) - 1):
                    log1_new = log1[(split_pos1[i] + 1):split_pos1[i + 1]]
                    log2_new = log2[(split_pos2[i] + 1):split_pos2[i + 1]]
                    sub_delimiters1 = {}
                    for key in delimiters1:
                        list_new = []
                        list_old = delimiters1[key]
                        for pos in list_old:
                            if (pos >= split_pos1[i] + 1 and pos < split_pos1[i + 1]):
                                list_new.append(pos - (split_pos1[i] + 1))
                        if len(list_new) > 0:
                            sub_delimiters1[key] = list_new
                    sub_delimiters2 = {}
                    for key in delimiters2:
                        list_new = []
                        list_old = delimiters2[key]
                        for pos in list_old:
                            if (pos >= split_pos2[i] + 1 and pos < split_pos2[i + 1]):
                                list_new.append(pos - (split_pos2[i] + 1))
                        if len(list_new) > 0:
                            sub_delimiters2[key] = list_new
                    sub_bracket1 = {}
                    for key in brackets1:
                        list_new = []
                        list_old = brackets1[key]
                        for tup in list_old:
                            if (tup[0] >= split_pos1[i] + 1 and tup[1] < split_pos1[i + 1]):
                                list_new.append([tup[0] - (split_pos1[i] + 1), tup[1] - (split_pos1[i] + 1)])
                        if len(list_new) > 0:
                            sub_bracket1[key] = list_new
                    sub_bracket2 = {}
                    for key in brackets2:
                        list_new = []
                        list_old = brackets2[key]
                        for tup in list_old:
                            if (tup[0] >= split_pos2[i] + 1 and tup[1] < split_pos2[i + 1]):
                                list_new.append([tup[0] - (split_pos2[i] + 1), tup[1] - (split_pos2[i] + 1)])
                        if len(list_new) > 0:
                            sub_bracket2[key] = list_new
                    newnode = Node(self.logIDL, log2_new, sub_delimiters2, sub_bracket2, self.cid)
                    newnode.parent = self
                    newnode.merge_log(logID, log1_new, sub_delimiters1, sub_bracket1)
                    self.children.append(newnode)
                self.addValue(log, logID, self.cid)
                return

            if (not delimiters_same_pattern):
                self.makeLeaf()
                self.addValue(log, logID, self.cid)
                return
            sim, delimiter_now = sim_log2log(log1, delimiters1, brackets1, log2, delimiters2, brackets2)
            if (delimiter_now == None):
                self.makeLeaf()
                self.addValue(log, logID, self.cid)
                return
            split_pos1 = pos1[delimiter_now].copy()
            split_pos2 = pos2[delimiter_now].copy()
            if (split_pos1[0] != 0):
                split_pos1.insert(0, -1)
            if (split_pos1[-1] != len(log1) - 1):
                split_pos1.append(len(log1))
            if (split_pos2[0] != 0):
                split_pos2.insert(0, -1)
            if (split_pos2[-1] != len(log2) - 1):
                split_pos2.append(len(log2))
            if (len(split_pos1) != len(split_pos2)):
                self.makeLeaf()
                self.addValue(log, logID, self.cid)
                return
            self.delimiter_now = delimiter_now

            self.pattern = ""
            if (split_pos2[0] == 0):
                self.pattern += delimiter_now
            for i in range(len(split_pos2) - 2):
                self.pattern += "F"
                self.pattern += delimiter_now
            self.pattern += "F"
            if (split_pos2[-1] == len(log2) - 1):
                self.pattern += delimiter_now

            for i in range(len(split_pos1) - 1):
                log1_new = log1[(split_pos1[i] + 1):split_pos1[i + 1]]
                log2_new = log2[(split_pos2[i] + 1):split_pos2[i + 1]]
                sub_delimiters1 = {}
                for key in delimiters1:
                    list_new = []
                    list_old = delimiters1[key]
                    for pos in list_old:
                        if (pos >= split_pos1[i] + 1 and pos < split_pos1[i + 1]):
                            list_new.append(pos - (split_pos1[i] + 1))
                    if len(list_new) > 0:
                        sub_delimiters1[key] = list_new
                sub_delimiters2 = {}
                for key in delimiters2:
                    list_new = []
                    list_old = delimiters2[key]
                    for pos in list_old:
                        if (pos >= split_pos2[i] + 1 and pos < split_pos2[i + 1]):
                            list_new.append(pos - (split_pos2[i] + 1))
                    if len(list_new) > 0:
                        sub_delimiters2[key] = list_new
                sub_bracket1 = {}
                for key in brackets1:
                    list_new = []
                    list_old = brackets1[key]
                    for tup in list_old:
                        if (tup[0] >= split_pos1[i] + 1 and tup[1] < split_pos1[i + 1]):
                            list_new.append([tup[0] - (split_pos1[i] + 1), tup[1] - (split_pos1[i] + 1)])
                    if len(list_new) > 0:
                        sub_bracket1[key] = list_new
                sub_bracket2 = {}
                for key in brackets2:
                    list_new = []
                    list_old = brackets2[key]
                    for tup in list_old:
                        if (tup[0] >= split_pos2[i] + 1 and tup[1] < split_pos2[i + 1]):
                            list_new.append([tup[0] - (split_pos2[i] + 1), tup[1] - (split_pos2[i] + 1)])
                    if len(list_new) > 0:
                        sub_bracket2[key] = list_new
                newnode = Node(self.logIDL, log2_new, sub_delimiters2, sub_bracket2, self.cid)
                newnode.parent = self
                newnode.merge_log(logID, log1_new, sub_delimiters1, sub_bracket1)
                self.children.append(newnode)
            self.addValue(log, logID, self.cid)
            return
        self.addValue(log, logID, self.cid)
        return


def flatten_all_poses(delimiters, brackets):
    poses = []
    poses1 = []
    poses2 = []
    for symbol in delimiters:
        poses1.append(delimiters[symbol])
    for symbol in brackets:
        poses2.append(brackets[symbol])

    poses1 = sum(poses1, [])
    poses2 = sum(poses2, [])
    poses2 = sum(poses2, [])
    poses.append(poses1)
    poses.append(poses2)
    poses = sum(poses, [])
    poses.sort()
    return poses


def is_Date(word):
    stop_word = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                 'November', 'December',
                 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
                 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    if word in stop_word:
        return True
    return False



class prefixTreeNode:
    def __init__(self, word, cid):
        self.word = word
        self.next = {}
        self.content = ""
        self.contents = []
        self.visited = 0
        self.cid = cid
        self.last = []

    def add_next(self, word):
        newNode = prefixTreeNode(word)
        self.next[word] = newNode
        newNode.last.append(self)


def addStringToPrefixTree(nodeStr, node_now):
    index = 0
    while (index < len(nodeStr)):
        word_now = nodeStr[index]
        if (word_now != '<'):
            if (word_now in node_now.next.keys()):
                node_now = node_now.next[word_now]
            else:
                node_now.add_next(word_now)
                node_now = node_now.next[word_now]
            index += 1
        elif (nodeStr[index:index + 3] == "<*>"):
            if ("<*>" in node_now.next.keys()):
                node_now = node_now.next["<*>"]
            else:
                node_now.add_next("<*>")
                node_now = node_now.next["<*>"]
            index = index + 3
        elif (nodeStr[index:index + 4] == "<\d>"):
            if ("<\d>" in node_now.next.keys()):
                node_now = node_now.next["<\d>"]
            else:
                node_now.add_next("<\d>")
                node_now = node_now.next["<\d>"]
            index = index + 4
        else:
            if (word_now in node_now.next.keys()):
                node_now = node_now.next[word_now]
            else:
                node_now.add_next(word_now)
                node_now = node_now.next[word_now]
            index += 1
    return node_now


class prefixTree:
    def __init__(self):
        self.root = prefixTreeNode("begin", -1)
        self.cid2node = {}

    def Str2List(self, nodeStr):
        StrList = []
        index = 0
        while (index < len(nodeStr)):
            word_now = nodeStr[index]
            if (word_now != '<'):
                StrList.append(word_now)
                index += 1
            elif (nodeStr[index:index + 3] == "<*>"):
                StrList.append("<*>")
                index = index + 3
            elif (nodeStr[index:index + 3] == "<->"):
                StrList.append("<->")
                index = index + 3
            elif (nodeStr[index:index + 4] == "<\d>"):
                StrList.append("<\d>")
                index = index + 4
            else:
                StrList.append("<")
                index += 1
        return StrList

    def addStr2PrefixTree(self, node_now, index, nodeStrList, cid):
        isBreak = False
        for i in range(len(node_now.contents)):
            if (index >= len(nodeStrList)):
                return index, node_now, i
            word = nodeStrList[index]
            ch = node_now.contents[i]
            if (word == ch):
                index += 1
            else:
                newNode = prefixTreeNode(ch, cid)
                newNode.next = node_now.next
                newNode.contents = node_now.contents[i + 1:]
                newNode.cid = node_now.cid
                for c in newNode.contents:
                    newNode.content = newNode.content + c
                for key in newNode.next.keys():
                    node_next = newNode.next[key]
                    if (node_now in node_next.last):
                        node_next.last.remove(node_now)
                        node_next.last.append(newNode)
                node_now.next = {}
                node_now.next[ch] = newNode
                newNode.last.append(node_now)
                if (node_now.cid != cid):
                    node_now.cid = -1
                node_now.contents = node_now.contents[:i]
                node_now.content = ""
                for c in node_now.contents:
                    node_now.content = node_now.content + c
                isBreak = True
                break
        if (not isBreak and node_now.cid != cid):
            node_now.cid = -1
        return index, node_now, -1

    def addNodeStrList2PrefixTree(self, node_now, nodeStrList, cid):
        index = 0
        while (index < len(nodeStrList)):
            word_now = nodeStrList[index]
            if (word_now in node_now.next.keys()):
                node_now = node_now.next[word_now]
                index += 1
                if (len(node_now.contents) > 0):
                    index, node_now, i = self.addStr2PrefixTree(node_now, index, nodeStrList, cid)
                    if (i != -1):
                        newNode = prefixTreeNode(node_now.contents[i], cid)
                        newNode.contents = node_now.contents[i + 1:]
                        for ch in newNode.contents:
                            newNode.content = newNode.content + ch
                        newNode.cid = node_now.cid
                        newNode.next = node_now.next
                        for key in newNode.next.keys():
                            node_next = newNode.next[key]
                            if (node_now in node_next.last):
                                node_next.last.remove(node_now)
                                node_next.last.append(newNode)

                        node_now.contents = node_now.contents[:i]
                        node_now.content = ""
                        if (node_now.cid != cid):
                            node_now.cid = -1
                        for ch in node_now.contents:
                            node_now.content = node_now.content + ch
                        node_now.next = {}
                        node_now.next[newNode.word] = newNode
                        newNode.last.append(node_now)
                else:
                    if (node_now.cid != cid):
                        node_now.cid = -1
            else:
                newNode = prefixTreeNode(word_now, cid)
                newNode.contents = nodeStrList[index + 1:]
                for ch in newNode.contents:
                    newNode.content = newNode.content + ch
                node_now.next[word_now] = newNode
                if (node_now.cid != cid):
                    node_now.cid = -1
                newNode.last.append(node_now)
                node_now = node_now.next[word_now]
                break
        return node_now

    def delete_prefix_tree(self, cid):
        nodeStack = []
        cidNode = self.cid2node[cid]
        nodeStack.append(cidNode)
        while (nodeStack):
            node_now = nodeStack.pop()
            if (node_now.cid == cid):
                for node_last in node_now.last:
                    if (node_now.word in node_last.next.keys()):
                        node_last.next.pop(node_now.word)
                        nodeStack.append(node_last)
                    if (len(node_last.next) == 1 and node_last.cid == -1 and node_last.word != "begin"):
                        key = ""
                        for k in node_last.next.keys():
                            key = k
                        if (node_last.next[key].word == "success"):
                            continue
                        if (node_last.next[key].cid != cid):
                            tmp = node_last.next[key]
                            node_last.contents.append(tmp.word)
                            node_last.contents = node_last.contents + tmp.contents
                            node_last.content = node_last.content + tmp.word + tmp.content
                            node_last.cid = tmp.cid
                            node_last.next = tmp.next
                            for key in node_last.next.keys():
                                nextNode = node_last.next[key]
                                nextNode.last.remove(tmp)
                                nextNode.last.append(node_last)
                del node_now
        self.cid2node.pop(cid)
        return

    def add_prefix_tree_with_templateTree_with_compress(self, nodeStr, cid):
        nodeStrList = self.Str2List(nodeStr)
        nodeStrL = []

        nodeStrL.append(nodeStrList)

        node_now = self.root
        for i in range(len(nodeStrL)):
            StrL = nodeStrL[i]
            node_now = self.addNodeStrList2PrefixTree(node_now, StrL, cid)

        newNode = prefixTreeNode("success", cid)
        node_now.next["success"] = newNode
        newNode.last.append(node_now)
        self.cid2node[cid] = newNode
        return

    def match(self, log, node_now, match_not_wildcard=False):
        index = 0
        node_now_matched = False
        while (index < len(log)):
            if (node_now.content != ""):
                index, match_index = match_Str(log, index, node_now.contents)
                node_now_matched = True
                if (index == -1):
                    return -1
                if (index >= len(log)):
                    if ("success" in node_now.next.keys() and match_index >= len(
                            node_now.contents) and match_not_wildcard):
                        return node_now.next["success"].cid
                    else:
                        return -1
                ch_now = log[index]
                if (ch_now == " "):
                    while (log[index + 1] == " "):
                        index += 1

                if (ch_now not in node_now.next.keys()):
                    if ((ch_now.isdigit() or ch_now == '-') and '<\d>' in node_now.next.keys()):
                        node_tmp = node_now.next["<\d>"]
                        if (ch_now == "-"):
                            index_ = index + 1
                        else:
                            index_ = index
                        ch_now_ = log[index_]
                        while (ch_now_.isdigit()):
                            index_ += 1
                            if (index_ >= len(log)):
                                break
                            ch_now_ = log[index_]
                        cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                        if (cid != -1):
                            return cid
                    if ('<*>' in node_now.next.keys()):
                        if (ch_now in node_now.next.keys()):
                            node_tmp = node_now.next[ch_now]
                            if (len(node_tmp.contents) != 0):
                                index_ = index + 1
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                        node_tmp = node_now.next["<*>"]
                        if (len(node_tmp.contents) == 0 and "success" in node_tmp.next.keys() and " " not in log[
                                                                                                             index:] and match_not_wildcard):
                            return node_tmp.next["success"].cid
                        elif (len(node_tmp.contents) != 0):
                            if (len(node_tmp.contents) > 0):
                                index_ = match_wildcard(log, index, node_tmp.contents[0])
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                        else:
                            for values in node_tmp.next.keys():
                                index_ = match_wildcard(log, index, values)
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                    else:
                        return -1
                else:
                    if ('<*>' in node_now.next.keys()):
                        if (ch_now in node_now.next.keys()):
                            node_tmp = node_now.next[ch_now]
                            if (len(node_tmp.contents) != 0):
                                index_ = index + 1
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                        node_tmp = node_now.next["<*>"]
                        if (len(node_tmp.contents) == 0 and "success" in node_tmp.next.keys() and " " not in log[
                                                                                                             index:] and match_not_wildcard):
                            return node_tmp.next["success"].cid
                        elif (len(node_tmp.contents) != 0):
                            if (len(node_tmp.contents) > 0):
                                index_ = match_wildcard(log, index, node_tmp.contents[0])
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                        else:
                            for values in node_tmp.next.keys():
                                index_ = match_wildcard(log, index, values)
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                    if ((ch_now == "-" or ch_now.isdigit()) and "<\d>" in node_now.next.keys()):
                        if (ch_now == "-"):
                            index_ = index + 1
                        else:
                            index_ = index
                        node_tmp = node_now.next["<\d>"]

                        ch_now_ = log[index_]
                        while (ch_now_.isdigit()):
                            index_ += 1
                            if (index_ >= len(log)):
                                break
                            ch_now_ = log[index_]
                        cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                        if (cid != -1):
                            return cid
                    node_now = node_now.next[ch_now]
                    node_now_matched = False
                    match_not_wildcard = True
                    index += 1
            else:
                ch_now = log[index]
                if (ch_now not in node_now.next.keys()):
                    if (ch_now.isdigit() and '<\d>' in node_now.next.keys()):
                        while (ch_now.isdigit()):
                            index += 1
                            if (index >= len(log)):
                                if ("success" in node_now.next.keys()):
                                    return node_now.next["success"].cid
                                else:
                                    return -1
                            ch_now = log[index]
                        node_now = node_now.next['<\d>']
                    elif ('<*>' in node_now.next.keys()):
                        node_now = node_now.next['<*>']
                        if (len(node_now.contents) > 0):
                            index = match_wildcard(log, index, node_now.contents[0])
                        else:
                            stopkey = " "
                            for key in node_now.next.keys():
                                if (not key.isalnum()):
                                    stopkey = key
                            index = match_wildcard(log, index, stopkey)
                    else:
                        return -1
                else:
                    if ('<*>' in node_now.next.keys()):
                        if (ch_now in node_now.next.keys()):
                            node_tmp = node_now.next[ch_now]
                            if (len(node_tmp.contents) != 0):
                                index_ = index + 1
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                        node_tmp = node_now.next["<*>"]
                        if (len(node_tmp.contents) == 0 and "success" in node_tmp.next.keys() and " " not in log[
                                                                                                             index:] and match_not_wildcard):
                            return node_tmp.next["success"].cid
                        elif (len(node_tmp.contents) != 0):
                            if (len(node_tmp.contents) > 0):
                                index_ = match_wildcard(log, index, node_tmp.contents[0])
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                        else:
                            for values in node_tmp.next.keys():
                                index_ = match_wildcard(log, index, values)
                                cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                                if (cid != -1):
                                    return cid
                    if (ch_now.isdigit() and "<\d>" in node_now.next.keys()):
                        node_tmp = node_now.next["<\d>"]
                        index_ = index
                        ch_now_ = ch_now
                        while (ch_now_.isdigit()):
                            index_ += 1
                            if (index_ >= len(log)):
                                break
                            ch_now_ = log[index_]
                        cid = self.match(log[index_:], node_tmp, match_not_wildcard)
                        if (cid != -1):
                            return cid
                    node_now = node_now.next[ch_now]
                    match_not_wildcard = True
                    index += 1
        if (len(node_now.contents) > 0 and log == ""):
            return -1
        if (not node_now_matched and len(node_now.contents) > 0):
            return -1
        if ("success" in node_now.next.keys()):
            return node_now.next["success"].cid
        else:
            return -1


def match_wildcard(log, index, stopdelimiter):
    while (index < len(log)):
        ch_now = log[index]
        if (ch_now != stopdelimiter and ch_now != " "):
            index += 1
        else:
            return index
    return len(log)


def match_Str(log, index, StringL):
    match_index = 0
    ch_index = index
    last_is_digit = False
    brackets_stack = []
    while (ch_index < len(log)):
        if (match_index >= len(StringL)):
            return ch_index, match_index
        ch_now = log[ch_index]
        if (ch_now == " "):
            while (log[ch_index + 1] == " "):
                ch_index += 1
        match_now = StringL[match_index]
        if (match_now == "<*>"):
            if (match_index + 1 >= len(StringL)):
                if (" " not in log[ch_index + 1:]):
                    return len(log), match_index + 1
                else:
                    return log[ch_index + 1:].find(" ") + ch_index + 1, match_index + 1
            match_next = StringL[match_index + 1]
            if ((
                    ch_now == '(' or ch_now == "[" or ch_now == '{') and match_next != '(' and match_next != '[' and match_next != '{'):
                brackets_stack.append(1)
            elif (ch_now == ')' or ch_now == "]" or ch_now == '}'):
                if (brackets_stack):
                    brackets_stack.pop()

            if (ch_now != match_next or brackets_stack):
                ch_index += 1
                continue
            else:
                ch_index += 1
                match_index += 2
        elif (match_now == "<\d>"):
            if (ch_now.isdigit() or (ch_now == "-" and last_is_digit == False)):
                ch_index += 1
                last_is_digit = True
                continue
            else:
                last_is_digit = False
                match_index += 1
        else:
            if (ch_now != match_now):
                return -1, match_index
            else:
                ch_index += 1
                match_index += 1
    if (last_is_digit):
        match_index += 1
    return len(log), match_index


class Template_tree:
    def __init__(self, tau, cid):
        self.logIDL = []
        self.nodeList = []
        self.pattern = ""
        self.sim_tau = tau
        self.min_sim = 1
        self.min_score_demand = tau
        self.String = ""
        self.cid = cid

        self.wildcards_seq = {cid: []}
        self.wildcards_num = 0
        self.wildcards_node = {}
        self.wildcards_values = {}
        self.wid = 0
        self.cid2template = {}

    def similarity_score(self, Split_words_bySpace, delimiters, brackets):
        if len(self.nodeList) != len(Split_words_bySpace):
            return 0
        length = len(self.nodeList)
        if (length > 1):
            sim_score = 1.0 / (length + 1)
            for i in range(length):
                sim_score += sim_log_node(self.nodeList[i], Split_words_bySpace[i], delimiters[i], brackets[i]) / (
                        length + 1)
        else:
            sim_score = sim_log_node(self.nodeList[0], Split_words_bySpace[0], delimiters[0], brackets[0])
        return sim_score

    def merge_log_node(self, logid, Split_words_bySpace, delimiters, brackets, sim_score):
        self.logIDL.append(logid)
        if sim_score < self.min_sim:
            self.min_sim = sim_score
        length = len(self.nodeList)
        for i in range(length):
            self.nodeList[i].merge_log(logid, Split_words_bySpace[i], delimiters[i], brackets[i])
        return

    def copy(self):
        newCluster = Template_tree(self.sim_tau, self.cid)
        newCluster.min_sim = self.min_sim
        newCluster.min_score_demand = self.min_score_demand
        newCluster.LCS = self.LCS
        newCluster.logIDL = self.logIDL.copy()
        newCluster.pattern = self.pattern
        for node in self.nodeList:
            newCluster.nodeList.append(node.copy())
        return newCluster

    def toString(self):
        templateStr = ""
        self.wildcards_node = {}
        self.wildcards_values = {}
        self.wid = 0

        node_id = 0
        for ch in self.pattern:
            if ch != "F":
                templateStr += ch
            else:
                node = self.nodeList[node_id]
                nodeStr = node.toString(self)
                if (nodeStr == ""):
                    templateStr = templateStr[0:-1]
                templateStr += nodeStr
                node_id += 1
        return templateStr

    def toTemplates(self):
        self.cid2template = {}
        template_str = self.toString()

        if (len(self.wildcards_node.keys()) == 0):
            cid = list(self.wildcards_seq.keys())[0]
            self.cid2template[cid] = template_str
        else:
            template_str_List = re.split("<\*>", template_str)
            for id in self.wildcards_seq.keys():
                seq = self.wildcards_seq[id]
                template = ""
                for i in range(len(template_str_List) - 1):
                    template = template + template_str_List[i] + seq[i]
                template += template_str_List[-1]
                self.cid2template[id] = template



class LogParser:

    def __init__(self, log_format, indir='./', outdir='./result/', st=0.75, isChinese=False, size=None,
                 printlenth=2000):
        self.path = indir
        self.st = st
        self.savePath = outdir
        self.df_log = None
        self.max_size = size
        self.log_format = log_format
        self.logClusters = {}
        self.invert_table = {}
        self.isChinese = isChinese
        self.logmessages = []
        self.clusters_candidate = []
        self.tolerance = 0.1
        self.prefix_tree = prefixTree()
        self.aa = printlenth
        self.logName = None

        self.max_cid = 0
        self.cidMap = {}

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='utf-8') as fin:
            if self.max_size:
                lines = fin.readlines()[:self.max_size]
            else:
                lines = fin.readlines()
            for line in lines:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def find_candidateClu(self, words_set):
        count = {}
        for i in range(len(words_set)):
            words = words_set[i]
            for word in words:
                if (word not in self.invert_table.keys()):
                    continue
                clusterIds = self.invert_table[word]
                for cid, position, score in clusterIds:
                    if (position == i):
                        if (not cid in count.keys()):
                            count[cid] = 0.0
                        count[cid] += score

        candidate_ids = []
        scores = []
        temp = sorted(count.items(), key=lambda x: x[1], reverse=True)
        for id, score in temp:
            if (score >= self.logClusters[id].min_score_demand):
                candidate_ids.append(id)
                scores.append(score)
        for cid in self.clusters_candidate:
            candidate_ids.append(cid)
            scores.append(0)
        return candidate_ids, scores

    def parse(self, logName):
        print('Parsing file in online mode: ' + os.path.join(self.path, logName))
        self.logName = logName
        self.load_data()

        time_parse = timer()
        ret = []

        for idx, line in self.df_log.iterrows():
            if idx % self.aa == 0:
                print('Processed {0:.1f}% of log lines.'.format(idx * 100.0 / len(self.df_log)))
                # self.update_Invert()
            lineID = line['LineId']
            logmessage = line['Content']
            logid = lineID - 1

            match_id = self.prefix_tree.match(logmessage, self.prefix_tree.root)
            if (match_id == -1):
                trivial_log, Split_words_bySpace, delimiters, brackets, hasDigit = SplitFirstLayer(logmessage)
                fold_nodes(trivial_log, Split_words_bySpace, delimiters, brackets, hasDigit)

                match_id, score = self.search_similar(trivial_log, Split_words_bySpace, delimiters, brackets)
                if match_id is None:
                    cid = self.max_cid
                    newCluster = Template_tree(self.st, cid)
                    newCluster.logIDL = [logid]
                    for i in range(len(Split_words_bySpace) - 1):
                        newCluster.pattern += "F "
                    newCluster.pattern += "F"
                    for i in range(len(Split_words_bySpace)):
                        newNode = Node([logid], Split_words_bySpace[i], delimiters[i], brackets[i], cid)
                        newCluster.nodeList.append(newNode)

                    newCluster.cid = cid
                    self.logClusters[cid] = newCluster
                    self.addClusterToInvertTable(newCluster, cid)
                    self.prefix_tree.add_prefix_tree_with_templateTree_with_compress(newCluster.toString(), cid)
                    self.cidMap[cid] = cid
                    self.max_cid += 1
                    ret.append(cid)
                else:
                    cluster = self.logClusters[match_id]
                    cluster.merge_log_node(logid, Split_words_bySpace, delimiters, brackets, score)

                    removed_nodes = []
                    for key in cluster.wildcards_node.keys():
                        node = cluster.wildcards_node[key]
                        if (node.removed):
                            while (node.removed):
                                node = node.parent
                            removed_nodes.append([key, node])
                    lasti = -100
                    lastnode = None
                    tmpid = []
                    ids = []
                    nodes = []
                    for i, node in removed_nodes:
                        if ((not (i == lasti + 1 and node == lastnode)) and tmpid):
                            nodes.append(lastnode)
                            ids.append(tmpid)
                            tmpid = []
                        tmpid.append(i)
                        lasti = i
                        lastnode = node
                    if (tmpid):
                        nodes.append(lastnode)
                        ids.append(tmpid)
                    rm_num=0
                    for cid_ in cluster.wildcards_seq.keys():
                        seq = cluster.wildcards_seq[cid_]
                        removed_num = 0
                        for i in range(len(ids)):
                            idL = ids[i]
                            for j in range(len(idL) - 1):
                                seq.pop(idL[0] - removed_num + 1)
                            removed_num += len(idL) - 1
                        rm_num=removed_num
                        cluster.wildcards_seq[cid_] = seq
                    cluster.wildcards_num-=rm_num
                    TreeStr = cluster.toString()

                    if (cluster.wildcards_num != len(cluster.wildcards_node)):
                        cluster.wildcards_num = len(cluster.wildcards_node)
                        index = 0
                        previous_seq = cluster.wildcards_seq
                        now_seq = []
                        for key in cluster.wildcards_node.keys():
                            node = cluster.wildcards_node[key]
                            if (node.becomeWildcard):
                                node.becomeWildcard = False
                                for cid in previous_seq.keys():
                                    seq = previous_seq[cid]
                                    seq.insert(index, node.firstWord)
                                    previous_seq[cid] = seq
                            now_seq.append(node.last)
                            index += 1
                        cid = self.max_cid
                        self.max_cid += 1
                        match_id = cid
                        self.cidMap[cid] = cid
                        cluster.wildcards_seq[cid] = now_seq
                    else:
                        now_seq = []
                        for key in cluster.wildcards_node.keys():
                            node = cluster.wildcards_node[key]
                            if (node.tooManyVals):
                                now_seq.append("<*>")
                            else:
                                now_seq.append(node.last)
                        match_id = -1
                        for id in cluster.wildcards_seq.keys():
                            if (now_seq == cluster.wildcards_seq[id]):
                                match_id = id
                                break
                        if (match_id == -1):
                            cid = self.max_cid
                            self.max_cid += 1
                            self.cidMap[cid] = cid
                            match_id = cid
                            cluster.wildcards_seq[cid] = now_seq

                    for i in cluster.wildcards_node.keys():
                        node = cluster.wildcards_node[i]
                        if (node.tooManyVals):
                            new_seqL = {}
                            for cid_ in cluster.wildcards_seq.keys():
                                seq = cluster.wildcards_seq[cid_]
                                seq[i] = "<*>"
                                if (seq not in new_seqL.values()):
                                    new_seqL[cid_] = seq
                                    self.cidMap[cid_] = cid_
                                else:
                                    key_ = cid_
                                    for key in new_seqL.keys():
                                        if (new_seqL[key] == seq):
                                            key_ = key
                                            break
                                    self.cidMap[cid_] = key_
                            cluster.wildcards_seq = new_seqL

                    preStr = cluster.cid2template.copy()
                    cluster.toTemplates()
                    prekeys = preStr.keys()

                    for key in cluster.cid2template.keys():
                        if (key not in prekeys):
                            if (re.search(r'\w', TreeStr)):
                                self.prefix_tree.add_prefix_tree_with_templateTree_with_compress(
                                    cluster.cid2template[key], key)
                        else:
                            if(preStr[key] != cluster.cid2template[key]):
                                if (key in self.prefix_tree.cid2node.keys()):
                                    self.prefix_tree.delete_prefix_tree(key)
                                if(re.search(r'\w', TreeStr)):
                                    self.prefix_tree.add_prefix_tree_with_templateTree_with_compress(
                                        cluster.cid2template[key], key)
                            preStr.pop(key)

                    for key in preStr.keys():
                        if (key in self.prefix_tree.cid2node.keys()):
                            self.prefix_tree.delete_prefix_tree(key)

                    ret.append(match_id)
            else:
                ret.append(match_id)

        time_end = timer()
        print("time:" + str(time_end - time_parse))

        ret=self.outputResult(self.logClusters, ret)

        return ret

    def update_Invert(self):
        self.invert_table = {}
        self.clusters_candidate = []
        for cid in self.logClusters.keys():
            self.addClusterToInvertTable(self.logClusters[cid], cid)

    def search_similar(self, trivial_log, Split_words_bySpace, delimiters, brackets):
        match_id = -1
        max_score = 0
        candidates, sameTkCount = self.find_candidateClu(trivial_log)
        for cid in candidates:
            cluster = self.logClusters[cid]
            score = cluster.similarity_score(Split_words_bySpace, delimiters, brackets)
            if score >= 0.999:
                return cid, score
            if (score > cluster.sim_tau):
                if score > max_score:
                    max_score = score
                    match_id = cid
        if match_id != -1:
            return match_id, max_score
        return None, None

    def addWordToInvertTable(self, word, cid, score, position):
        invert_table = self.invert_table
        if (word not in invert_table.keys()):
            invert_table[word] = []
        find = False
        for tuple3 in invert_table[word]:
            if (tuple3[0] == cid and tuple3[1] == position):
                find = True
                if (score > tuple3[2]):
                    invert_table[word].remove(tuple3)
                    invert_table[word].append([cid, position, score])
        if (not find):
            invert_table[word].append([cid, position, score])

    def addClusterToInvertTable(self, newCluster, cid):
        leafNode_candidate = []
        in_inv = 0
        scoreFirstLayer = 1.0 / (len(newCluster.nodeList) + 1)
        newCluster.min_score_demand = newCluster.sim_tau - scoreFirstLayer

        for i in range(len(newCluster.nodeList)):
            leafNode_candidate.append([newCluster.nodeList[i], scoreFirstLayer, i])
        while (leafNode_candidate):
            [node, score, position] = leafNode_candidate.pop(0)
            if (node.children):
                score_this_layer = score * (1.0 / (len(node.children) + 1))
                newCluster.min_score_demand -= score_this_layer
                for child in node.children:
                    leafNode_candidate.append([child, score_this_layer, position])
            else:
                if (len(node.values) > 1 or node.tooManyVals):
                    if (node.hasDigit):
                        newCluster.min_score_demand -= 0.5 * score
                    continue
                elif (not node.delimiters and not node.brackets and node.word.isalpha()):
                    self.addWordToInvertTable(node.word, cid, score, position)
                    in_inv = 1
                else:
                    newCluster.min_score_demand -= score
        if (newCluster.min_score_demand < 0 or in_inv == 0):
            self.clusters_candidate.append(cid)
        return


    def outputResult(self, logClustL, ret):
        templates = []
        filename = self.logName
        df_event = []

        cid2template = {}
        for cid in self.logClusters.keys():
            cluster = self.logClusters[cid]
            template_str = cluster.toString()
            if (len(cluster.wildcards_node.keys()) == 0):
                cid2template[cid] = template_str
            else:
                template_str_List = re.split("<\*>", template_str)
                for id in cluster.wildcards_seq.keys():
                    seq = cluster.wildcards_seq[id]
                    template = ""
                    for i in range(len(template_str_List) - 1):
                        template = template + template_str_List[i] + seq[i]
                    template += template_str_List[-1]
                    cid2template[id] = template

        for id in self.cidMap.keys():
            cid = id
            value = self.cidMap[cid]
            while (cid != value):
                cid = value
                value = self.cidMap[cid]
            self.cidMap[id] = value

        result = []
        outputMap = {}
        outid = 0
        count = {}
        for cid in ret:
            template = cid2template[self.cidMap[cid]]
            if (self.cidMap[cid] in outputMap.keys()):
                finalId = outputMap[self.cidMap[cid]]
                count[self.cidMap[cid]] += 1
            else:
                outputMap[self.cidMap[cid]] = outid
                finalId = outid
                count[self.cidMap[cid]] = 1
                outid += 1
            result.append(finalId)
            templates.append(template)

        print('templates count :', len(cid2template.keys()))

        for cid in cid2template.keys():
            eid = outputMap[cid]
            template_str = cid2template[cid]
            occur = count[cid]
            df_event.append([eid, template_str, occur])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = result
        self.df_log['EventTemplate'] = templates
        self.df_log.to_csv(os.path.join(self.savePath, filename + '_structured.csv'), index=False,
                           encoding="utf-8")
        df_event.to_csv(os.path.join(self.savePath, filename + '_templates.csv'), index=False, encoding="utf-8")

        return result
