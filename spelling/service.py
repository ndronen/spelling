from __future__ import print_function
import sys
import json
import requests
import xmltodict
import collections

def correct(url, text):
    body = {
            'sent-split': False, 'sent-start': False,
            'sent-end': False, 'case-sensitive': True,
            'text': text
            }

    response = requests.post(url, data=json.dumps(body))

    def http_error(response):
        return "HTTP error %d %s %s" % (
            response.status_code,
            response.reason,
            response.text)

    def service_error(response):
        return "Service error %s" % (response.text)

    if int(response.status_code) != 200:
        raise RuntimeError(http_error(response))

    response_dict = xmltodict.parse(response.text)
    spellcheck = response_dict['pktServiceResponse']['spellCheck']

    corrections = collections.defaultdict(list)
    wordlist = spellcheck['content']['wordList']

    if wordlist is not None:
        for word,values in spellcheck['content']['wordList'].items():
            original_word = values['originalWord']
            for replacement in values['replacement']:
                corrections['Context'].append(text)
                corrections['NonWord'].append(original_word)
                corrections['Candidate'].append(replacement['#text']),
                corrections['NgramRatio'].append(replacement['@ngram-ratio'])

    return corrections
