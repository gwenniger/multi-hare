#!/usr/bin/env python2.7

""" This file was adapted from code taken from Joan Puigcerver
    https://github.com/jpuigcerver/Laia
    for this file only the MIT license applies.

# The MIT License (MIT)
#
# Copyright (c) 2016 Joan Puigcerver, Daniel Martín-Albo and Mauricio Villegas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import re
import sys



# This is a custom modification of the TreebankWordTokenizer.
# The original tokenizer replaced double quotes with `` and ''.
# Since do not want to add new symbols to the sentences, we keep
# the double quotes symbols.
# This also makes trivial the function span_tokens, which
# TreebakWordTokenizer did not have implemented.
# Adapted from: http://www.nltk.org/_modules/nltk/tokenize/treebank.html
class CustomTreebankWordTokenizer:
    #starting quotes
    STARTING_QUOTES = [
        (re.compile(r'([ (\[{<])"'), r'\1 " '),
    ]

    #punctuation
    PUNCTUATION = [
        (re.compile(r'[;@#$%&.,/\u20AC$-]'), r' \g<0> '),
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
        (re.compile(r'[?!]'), r' \g<0> '),
        (re.compile(r"([^'])' "), r"\1 ' "),
    ]

    #parens, brackets, etc.
    PARENS_BRACKETS = [
        (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> '),
        (re.compile(r'--'), r' -- '),
    ]

    #ending quotes
    ENDING_QUOTES = [
        (re.compile(r'"'), ' " '),              # This line changes: do not replace "
        (re.compile(r'(\S)(\'\')'), r'\1 \2 '),
    ]

    CONTRACTIONS = [
        (re.compile(r"([^' ]')([^' ])"), r'\1 \2'),
    ]

    def tokenize(self, text):
        text = str(text)
        for regexp, substitution in self.STARTING_QUOTES:
            text = regexp.sub(substitution, text, re.UNICODE)

        for regexp, substitution in self.PUNCTUATION:
            text = regexp.sub(substitution, text, re.UNICODE)

        for regexp, substitution in self.PARENS_BRACKETS:
            text = regexp.sub(substitution, text, re.UNICODE)

        #add extra space to make things easier
        text = " " + text + " "

        for regexp, substitution in self.ENDING_QUOTES:
            text = regexp.sub(substitution, text, re.UNICODE)

        for regexp, substitution in self.CONTRACTIONS:
            text = regexp.sub(substitution, text, re.UNICODE)

        tokens = []
        for tok in text.split():
            if not re.match(r'^[A-Z0-9]+$', tok, re.UNICODE):
                tokens.append(tok)
            else:
                for t in re.split(r'([A-Z0-9])', tok, re.UNICODE):
                    if len(t) > 0: tokens.append(t)

        return tokens

    def span_tokens(self, text):
        text = str(text)
        tokens = self.tokenize(text)
        spans = []
        i = 0
        for tok in tokens:
            spans.append((i, i + len(tok)))
            i += len(tok)
            if i < len(text) and re.match(r'^\s$', text[i], re.UNICODE):
                i += 1
        return spans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type=argparse.FileType('r'), nargs='?', default=sys.stdin,
        help='input text file')
    parser.add_argument(
        'output', type=argparse.FileType('w'), nargs='?', default=sys.stdout,
        help='output text file')
    parser.add_argument(
        '--write-boundaries', type=argparse.FileType('w'), default=None,
        help='write token boundaries to this file')
    parser.add_argument(
        '--boundary', type=str, default='\\s',
        help='use this token as the boundary token')
    args = parser.parse_args()

    tokenizer = CustomTreebankWordTokenizer()
    lexicon = {}
    for line in args.input:
        line = re.sub(r'\s+', ' ', line.strip().decode('utf-8'), re.UNICODE)
        spans = tokenizer.span_tokens(line)
        tokens = map(lambda x: line[x[0]:x[1]], spans)
        args.output.write((u' '.join(tokens) + u'\n').encode('utf-8'))
        if args.write_boundaries is not None:
            for i in range(len(tokens)):
                pron = [args.boundary, tokens[i], args.boundary]
                if i > 0 and spans[i][0] == spans[i - 1][1]:
                    pron = pron[1:]
                if i < len(tokens) - 1 and spans[i][1] == spans[i + 1][0]:
                    pron = pron[:-1]
                pron = tuple(pron)
                if tokens[i] not in lexicon: lexicon[tokens[i]] = {}
                if pron not in lexicon[tokens[i]]: lexicon[tokens[i]][pron] = 1
                else: lexicon[tokens[i]][pron] += 1

    if args.write_boundaries:
        lexicon = lexicon.items()
        lexicon.sort()
        for (token, prons) in lexicon:
            for pron, cnt in prons.iteritems():
                args.write_boundaries.write((u'%s\t%d\t%s\n' % (token, cnt, ' '.join(pron))).encode('utf-8'))
        args.write_boundaries.close()

    args.output.close()