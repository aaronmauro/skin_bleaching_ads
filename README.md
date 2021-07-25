# Drum and Ebony Skin Bleaching Ads

This is a data analysis of skin bleaching ads from Ebony and Drum magazines over several decades. The data was collected by hand and captures several aspects of the ads, including the product, the active ingredient, the claims made, as well as the catch phrases used.

## Table of Contents

* [Data Structure](./#data)
* [Drum Most Frequent Words](./#drum_mfw)
* [Drum Ngrams](#drum_ngram)
* [Ebony Most Frequent Words](./#ebony_mfw)
* [Ebony Ngrams](#ebony_ngram)
* [Term Counts by Year](./#terms_by_year)
* [Parts of Speech](./#pos)

### To do

* compare pos and mfw over time between pubs
* plot ngrams in Drum to request images (ugly, hollywood, husband/bride/marriage)
* send list of 12-15 needed issues and ads


```python
__author__ = "Aaron Mauro"
__role__ = "researcher"
__institution__ = "Brock University"
__department__ = "Centre for Digital Humanities"
__email__ = "amauro@brocku.ca"
__status__ = "prototype/experiment"
__version__ = "0.1"
```


```python
# Imports

import os
import sys
import csv
import string
import itertools
import nltk
import gensim
import pprint as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import networkx as nx

# direct imports
from gensim import corpora, models
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from IPython.display import SVG, display

# REMOVE IN PROD
import pygal
#pd.set_option('display.mpl_style', 'default')
```


```python
# Select Matplotlib style
plt.matplotlib.style.use("seaborn")
```


```python
# English stopwords
STOPS = stopwords.words('english')
print(STOPS)
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]



```python
# Add additional stops
new_stops = ["I",] # add new stops here
for stop in new_stops:
    STOPS.append(stop)
```

<span id="data" class="anchor"></span>
## Data Structure and Sample Contents


```python
# Create dataframe from Ebony skin bleaching adds from 1960-1990

ebony = pd.read_csv("../Ebony.skin.ads.1960-1990.csv", parse_dates=True, index_col='Year')
ebony.head() # show first five rows of dataframe
#ebony.tail() # show last five rows of dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Source</th>
      <th>Product name</th>
      <th>chemical/active ingredient</th>
      <th>Structure</th>
      <th>Chemistry</th>
      <th>Claims</th>
      <th>Legal issues and Politics</th>
      <th>Race</th>
      <th>Age</th>
      <th>Advertising strategy *quotes-catch phrase*</th>
      <th>Size of Advert</th>
      <th>Pg reference (quentin marked)</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>Ebony</td>
      <td>Long Aid Bleach and Glow</td>
      <td>unnamed</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>" - wakes up dark, dull complexion! Conceals u...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1/2 pg</td>
      <td>63</td>
      <td>small part of 1.2 pg ad for Long Aid hair prod...</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mercolized Wax Cream</td>
      <td>ammoniated mercury; zinc oxide</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"If your skin doesn't look actually lighter af...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Mercolized Wax Cream guarantees lighter looki...</td>
      <td>1/4 pg</td>
      <td>72</td>
      <td>ingredient on image of product; not mentioned ...</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Black and White Bleaching Cream</td>
      <td>unnamed</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"And you, too, can have a glamorous complexion...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Lighter, brighter skin is irresistable"</td>
      <td>1/8 pg</td>
      <td>83</td>
      <td>drawing of white man and white woman in ad</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Nadinola Bleaching Cream</td>
      <td>"wonder-working A-M"</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Don't let dull, dark skin rob you of romance....</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"LIFE IS MORE FUN when your complexion is clea...</td>
      <td>full pg</td>
      <td>91</td>
      <td>two types advertised - oily and dry skin</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dr. Fred Palmer's Double Strength Skin Whitener</td>
      <td>zinc phenolsulfonate</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Yes in just 7 days be delighted how fast and ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"DR. FRED PALMER'S IN JUST 7 DAYS MUST GIVE YO...</td>
      <td>1/8 pg</td>
      <td>108</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create dataframe from Drum skin bleaching adds from 1965-1988

drum = pd.read_csv("../Drum_Skin_Lighteners_1965_1988.csv", parse_dates=True, index_col='Year')
drum.head() # show first five rows of dataframe
# drum.tail() # show last five rows of dataframe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product name</th>
      <th>chemical/active ingredient</th>
      <th>Claims</th>
      <th>Legal issues and Politics</th>
      <th>Race</th>
      <th>Age</th>
      <th>Advertising strategy *quotes-catch phrase*</th>
      <th>Size of Advert</th>
      <th>Pg. reference ( marked)</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1965-01-01</th>
      <td>ARTRA skin tone cream</td>
      <td>Hydroquinone</td>
      <td>…to make their skin lighter and lovelier…lovel...</td>
      <td>Black model and white pharmacist/doctor</td>
      <td>Black</td>
      <td>20+</td>
      <td>Lighter, lovelier skin today…the American way!'</td>
      <td>full pg.</td>
      <td>pg. 2</td>
      <td>The ad says that the cream was developed after...</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>brightens skin.' '…lightens from the first day...</td>
      <td>n/a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cream your skin lighter and brighter with ama...</td>
      <td>NaN</td>
      <td>pg. 2</td>
      <td>Ad states that it is a medicated beauty bar. S...</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>immediately.' '…keeps skin beautiful and clean...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>Artra beauty bar</td>
      <td>Hydroquinone</td>
      <td>…mild and gentle…keeps skin free from blemishe...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20+</td>
      <td>Medicated soap for complexion care</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>Aloma Crème blanche</td>
      <td>unnamed</td>
      <td>…clears and lightens the skin, smooth's away b...</td>
      <td>Black model</td>
      <td>Black</td>
      <td>20-35</td>
      <td>good things happen to a pretty girl</td>
      <td>full pg.</td>
      <td>NaN</td>
      <td>Ad states that using this product will increas...</td>
    </tr>
  </tbody>
</table>
</div>



## Preprocessing Ad Catch Phrases and Claims
As a first step in this analysis, we will capture the most frequent words (MFW) represented in both the catch phrases and claims of the ads. 


```python
# Set the Ebony dataframe index to the year
years_ebony = ebony.index
years_ebony = set(years_ebony.year)
```


```python
# Set the Drum dataframe index to the year
years_drum = drum.index
years_drum = set(years_drum.year)
```


```python
# Grab Ebony catch phrases by year and drop rows without data
catch_phrase_ebony = ebony["Advertising strategy *quotes-catch phrase*"].dropna()
```


```python
# Show first five rows of Ebony catch phrases
catch_phrase_ebony.head()
```




    Year
    1960-01-01    "Mercolized Wax Cream guarantees lighter looki...
    1960-01-01             "Lighter, brighter skin is irresistable"
    1960-01-01    "LIFE IS MORE FUN when your complexion is clea...
    1960-01-01    "DR. FRED PALMER'S IN JUST 7 DAYS MUST GIVE YO...
    1960-01-01    "Egyptian formula BLEACH CRÈME gives amazing r...
    Name: Advertising strategy *quotes-catch phrase*, dtype: object




```python
# Grab Drum catch phrases by year and drop rows without data
catch_phrase_drum = drum["Advertising strategy *quotes-catch phrase*"].dropna()
```


```python
# Show first five rows of Ebony catch phrases
catch_phrase_drum.head()
```




    Year
    1965-01-01      Lighter, lovelier skin today…the American way!'
    1965-01-01     Cream your skin lighter and brighter with ama...
    1965-01-01                   Medicated soap for complexion care
    1965-01-01                  good things happen to a pretty girl
    1965-01-01                                            See notes
    Name: Advertising strategy *quotes-catch phrase*, dtype: object




```python
# Grab Drum and Ebony claims
claims_ebony = ebony["Claims"].dropna()
claims_drum = drum['Claims'].dropna()
print("Ebony claims",claims_ebony.head())
print(claims_ebony.head())
```

    Ebony claims Year
    1960-01-01    " - wakes up dark, dull complexion! Conceals u...
    1960-01-01    "If your skin doesn't look actually lighter af...
    1960-01-01    "And you, too, can have a glamorous complexion...
    1960-01-01    "Don't let dull, dark skin rob you of romance....
    1960-01-01    "Yes in just 7 days be delighted how fast and ...
    Name: Claims, dtype: object
    Year
    1960-01-01    " - wakes up dark, dull complexion! Conceals u...
    1960-01-01    "If your skin doesn't look actually lighter af...
    1960-01-01    "And you, too, can have a glamorous complexion...
    1960-01-01    "Don't let dull, dark skin rob you of romance....
    1960-01-01    "Yes in just 7 days be delighted how fast and ...
    Name: Claims, dtype: object


<span id="drum_mfw" class="anchor"></span>
## Most Frequent Words

In an effort to collect keywords for later processes, we will grab the most frequent words (MFW) and compare them to most common bigram pairs. We'll start with Drum then do the same operations for Ebony.

## Drum Magaine MFW


```python
# Convert dataframe into a list of lists of sentences for Drum
drum_catch_phrase_and_claims = pd.merge(catch_phrase_drum, claims_drum, right_index=True, left_index=True).dropna()
drum_claims_phrase_list = drum_catch_phrase_and_claims.values.tolist()
drum_claims_phrase_list[:10]
```




    [["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and"],
     ["Lighter, lovelier skin today…the American way!'",
      "brightens skin.' '…lightens from the first day.' '…vanishes into skin instantly…starts working, starts lightening and brightening your skin"],
     ["Lighter, lovelier skin today…the American way!'",
      "immediately.' '…keeps skin beautiful and clean, makes it smooth and lovely"],
     ["Lighter, lovelier skin today…the American way!'",
      '…mild and gentle…keeps skin free from blemishes and pimples'],
     ["Lighter, lovelier skin today…the American way!'",
      "…clears and lightens the skin, smooth's away blemishes and spots, softens the skin'"],
     ["Lighter, lovelier skin today…the American way!'",
      '...gets to work immediately- to give you a lighter, lovelier complexion'],
     ["Lighter, lovelier skin today…the American way!'",
      "a fair complexion, clear skin that is blemish free and spot free can be obtained using this product'"],
     ["Lighter, lovelier skin today…the American way!'",
      "your skin will grow lighter, complexion smoother and blemish free.' '…it reveals the true, light loveliness of your skin'"],
     ["Lighter, lovelier skin today…the American way!'",
      "you will be more attractive, more desirable!' '…will admire your clear, light complexion.'"],
     ["Lighter, lovelier skin today…the American way!'",
      "clears complexion in a few days…' '…any blemishes and spots…Karroo creams gets rid of them…'"]]




```python
# Tokenize and preprocess lists of sentences
drum_word_corpus_list = []
for ad in drum_claims_phrase_list:
    sents = [ch for ch in " ".join(ad).lower() if ch not in string.punctuation+"…"]
    drum_word_corpus_list.append("".join(sents).split())
print(drum_word_corpus_list[:5])
```

    [['lighter', 'lovelier', 'skin', 'todaythe', 'american', 'way', 'to', 'make', 'their', 'skin', 'lighter', 'and', 'lovelierlovelier', 'and', 'lightera', 'little', 'more', 'every', 'day', 'american', 'scientist', 'made', 'artralightens', 'and'], ['lighter', 'lovelier', 'skin', 'todaythe', 'american', 'way', 'brightens', 'skin', 'lightens', 'from', 'the', 'first', 'day', 'vanishes', 'into', 'skin', 'instantlystarts', 'working', 'starts', 'lightening', 'and', 'brightening', 'your', 'skin'], ['lighter', 'lovelier', 'skin', 'todaythe', 'american', 'way', 'immediately', 'keeps', 'skin', 'beautiful', 'and', 'clean', 'makes', 'it', 'smooth', 'and', 'lovely'], ['lighter', 'lovelier', 'skin', 'todaythe', 'american', 'way', 'mild', 'and', 'gentlekeeps', 'skin', 'free', 'from', 'blemishes', 'and', 'pimples'], ['lighter', 'lovelier', 'skin', 'todaythe', 'american', 'way', 'clears', 'and', 'lightens', 'the', 'skin', 'smooths', 'away', 'blemishes', 'and', 'spots', 'softens', 'the', 'skin']]



```python
# Collect all words into single list
drum_word_corpus = []
for sent in drum_word_corpus_list:
    for word in sent:
        drum_word_corpus.append(word)
```


```python
# Total words in corpus and preview Drum word list 
drum_num = len(drum_word_corpus)
print(f"There are {drum_num:,} words in the Drum Magazine corpus.")
print(drum_word_corpus[:10])
```

    There are 211,112 words in the Drum Magazine corpus.
    ['lighter', 'lovelier', 'skin', 'todaythe', 'american', 'way', 'to', 'make', 'their', 'skin']



```python
# Create an NLTK Text object
drum_text_object = nltk.Text(drum_word_corpus)
```


```python
# Generate a sample concordance, displaying 120 characters of 25 matches
drum_text_object.concordance("good",120,25)
```

    Displaying 25 of 428 matches:
    e sooner you startexpect a lovely clear bright complexion good things happen to a pretty girl to make their skin lighter
     more every day american scientist made artralightens and good things happen to a pretty girl brightens skin lightens fr
    tarts working starts lightening and brightening your skin good things happen to a pretty girl immediately keeps skin bea
    keeps skin beautiful and clean makes it smooth and lovely good things happen to a pretty girl mild and gentlekeeps skin 
    mild and gentlekeeps skin free from blemishes and pimples good things happen to a pretty girl clears and lightens the sk
    he skin smooths away blemishes and spots softens the skin good things happen to a pretty girl gets to work immediately t
    ork immediately to give you a lighter lovelier complexion good things happen to a pretty girl a fair complexion clear sk
    ish free and spot free can be obtained using this product good things happen to a pretty girl your skin will grow lighte
    sh free it reveals the true light loveliness of your skin good things happen to a pretty girl you will be more attractiv
    ve more desirable will admire your clear light complexion good things happen to a pretty girl clears complexion in a few
    ays any blemishes and spotskarroo creams gets rid of them good things happen to a pretty girl the sooner you startexpect
    use ambi absolutely safeno burning or irritation its even good for painful sunburnt skin successful people use ambi skin
    ou light absolutely safeno burning or irritation its even good for painful sunburnt skin karroo morning karoo night make
    ou light absolutely safeno burning or irritation its even good for painful sunburnt skin karroo morning karoo night make
    o creams absolutely safeno burning or irritation its even good for painful sunburnt skin perfect skin beauty can be your
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone hilite
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone bright
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone karroo
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone karroo
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone hilite
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone bright
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone karroo
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone perfec
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone for th
    ning and smoothing your complexion as soon as its used is good and works so fast because it contains hydroquinone karroo



```python
# Create concordance object as list
drum_concord_list_obj_good = drum_text_object.concordance_list("good")
```


```python
# Unpack concordance list object into flat list
drum_concord_list_good = []
for con in drum_concord_list_obj_good:
    sent = "".join(list(con[4:]))
    drum_concord_list_good.append(sent)
```


```python
# Create frequence distribution
drum_concord_list_good_freqs = nltk.FreqDist(drum_concord_list_good)
```


```python
# Ten most common claims with the word 'good'
drum_concord_list_good_freqs.most_common(10)
```




    [('ur complexion as soon as its used isand works so fast because it containur complexion as soon as its used is good and works so fast because it contain',
      10),
     ('afeno burning or irritation its evenfor painful sunburnt skin karroo morafeno burning or irritation its even good for painful sunburnt skin karroo mor',
      2),
     ('ect a lovely clear bright complexionthings happen to a pretty girl to maect a lovely clear bright complexion good things happen to a pretty girl to ma',
      1),
     ('can scientist made artralightens andthings happen to a pretty girl brighcan scientist made artralightens and good things happen to a pretty girl brigh',
      1),
     ('lightening and brightening your skinthings happen to a pretty girl immedlightening and brightening your skin good things happen to a pretty girl immed',
      1),
     ('and clean makes it smooth and lovelythings happen to a pretty girl mild and clean makes it smooth and lovely good things happen to a pretty girl mild ',
      1),
     ('skin free from blemishes and pimplesthings happen to a pretty girl clearskin free from blemishes and pimples good things happen to a pretty girl clear',
      1),
     ('blemishes and spots softens the skinthings happen to a pretty girl gets blemishes and spots softens the skin good things happen to a pretty girl gets ',
      1),
     ('ve you a lighter lovelier complexionthings happen to a pretty girl a faive you a lighter lovelier complexion good things happen to a pretty girl a fai',
      1),
     ('e can be obtained using this productthings happen to a pretty girl your e can be obtained using this product good things happen to a pretty girl your ',
      1)]




```python
# Remove stop words
drum_word_corpus_no_stops = []
for word in drum_word_corpus:
    if word not in STOPS: #add extra stops above
        drum_word_corpus_no_stops.append(word)
drum_freqs = nltk.FreqDist(drum_word_corpus_no_stops)
drum_most_common_counts = drum_freqs.most_common(75)
drum_most_common = []
for t,c in drum_most_common_counts:
    drum_most_common.append(t)
pp.pprint(drum_most_common_counts)
```

    [('skin', 11413),
     ('complexion', 3707),
     ('light', 3321),
     ('karroo', 2997),
     ('lighter', 2685),
     ('ambi', 2671),
     ('beautiful', 2646),
     ('lovely', 2262),
     ('beauty', 2006),
     ('makes', 1950),
     ('extra', 1888),
     ('cream', 1826),
     ('use', 1804),
     ('new', 1757),
     ('look', 1701),
     ('smooth', 1669),
     ('lightening', 1650),
     ('fast', 1619),
     ('night', 1594),
     ('lightens', 1534),
     ('keeps', 1532),
     ('clear', 1363),
     ('people', 1332),
     ('smoother', 1173),
     ('pimples', 1149),
     ('spots', 1134),
     ('creams', 1090),
     ('great', 1067),
     ('hollywood', 1048),
     ('day', 992),
     ('blemishes', 977),
     ('looking', 904),
     ('successful', 884),
     ('morning', 881),
     ('dark', 859),
     ('seven', 835),
     ('miss', 805),
     ('acting', 803),
     ('ugly', 787),
     ('best', 778),
     ('lotion', 748),
     ('lighten', 741),
     ('queen', 712),
     ('men', 678),
     ('quick', 676),
     ('super', 658),
     ('lovelier', 655),
     ('time', 639),
     ('treatment', 623),
     ('face', 620),
     ('always', 614),
     ('karoo', 604),
     ('way', 599),
     ('hilite', 595),
     ('clearer', 576),
     ('works', 576),
     ('removes', 549),
     ('action', 544),
     ('fantastic', 531),
     ('thousands', 519),
     ('days', 518),
     ('says', 503),
     ('results', 472),
     ('woman', 465),
     ('modern', 449),
     ('healthy', 444),
     ('soft', 440),
     ('rose', 431),
     ('good', 428),
     ('give', 394),
     ('powerful', 382),
     ('neck', 378),
     ('one', 371),
     ('make', 365),
     ('made', 365)]



```python
# The most frequent words in Drum Skin Bleaching ads
drum_freqs.plot(35, title="Drum Magazine MFW")
```


    
![png](output_29_0.png)
    





    <AxesSubplot:title={'center':'Drum Magazine MFW'}, xlabel='Samples', ylabel='Counts'>




```python
# Looking at the long tail of MFW, people and success are related to lighter skin
# Example concordance
drum_text_object.concordance("people",120,25)
```

    Displaying 25 of 1332 matches:
    smoother lighter lovelier skin todaythe american way for people that want a lighter smoother complexion makes skin fair 
    our skin lighter and brighter with amazing new artra for people that want a lighter smoother complexion makes skin fair 
    arer and smoother medicated soap for complexion care for people that want a lighter smoother complexion makes skin fair 
     extra double strength skin lightening cream for men for people that want a lighter smoother complexion makes skin fair 
     will be clearer and smoother ambi extra always wins for people that want a lighter smoother complexion makes skin fair 
    g cream the double strength skin cream for smart men for people that want a lighter smoother complexion makes skin fair 
    smoother the best skin lightening cream in the world for people that want a lighter smoother complexion makes skin fair 
    ther you too can look as lovely as a beautiful bride for people that want a lighter smoother complexion makes skin fair 
    sire skin tone cream the beautiful brides complexion for people that want a lighter smoother complexion makes skin fair 
    er quote watch how quickly your skin becomes lighter for people that want a lighter smoother complexion makes skin fair 
    morning karoo night keeps you lovely keeps you light for people that want a lighter smoother complexion makes skin fair 
     extra double strength skin lightening cream for men for people that want a lighter smoother complexion makes skin fair 
     will be clearer and smoother ambi extra always wins for people that want a lighter smoother complexion makes skin fair 
    g cream the double strength skin cream for smart men for people that want a lighter smoother complexion makes skin fair 
    smoother the best skin lightening cream in the world for people that want a lighter smoother complexion makes skin fair 
     be won in the big artra skin tone cream competition for people that want a lighter smoother complexion makes skin fair 
    earer and smoother be lighter and lovelier with ambi for people that want a lighter smoother complexion makes skin fair 
    her ambi specialthe best skin lightener in the world for people that want a lighter smoother complexion makes skin fair 
    ou light protects skin against harsh sunlight successful people use ambi lighter clearer complexion with ambi ambi can g
    u the smooth clear lighter skin youve notices successful people have successful people use ambi scientifically formulate
    ter skin youve notices successful people have successful people use ambi scientifically formulatedclear the skin of pimp
    ake it smoother and lighter within a few days successful people use ambi absolutely safeno burning or irritation its eve
    ation its even good for painful sunburnt skin successful people use ambi skin looked lighter and smoother protects your 
    her protects your skin against harsh sunlight successful people use ambi worried by little pimples and spotskarroo takes
     pimples and spotskarroo takes these away too successful people use ambi keep my complexion looking light clear and smoo



```python
# Most common words in context for with keyword 'people'
drum_concord_list_obj = drum_text_object.concordance_list("people", 50)
drum_concord_list = []
for con in drum_concord_list_obj:
    sent = "".join(list(con[4:]))
    drum_concord_list.append(sent)
drum_concord_list_freqs = nltk.FreqDist(drum_concord_list)
drum_concord_list_freqs.most_common(10)
```




    [('ing cream for men forthat want a lighter sing cream for men for people that want a lighter s',
      2),
     ('extra always wins forthat want a lighter sextra always wins for people that want a lighter s',
      2),
     ('eam for smart men forthat want a lighter seam for smart men for people that want a lighter s',
      2),
     ('ream in the world forthat want a lighter sream in the world for people that want a lighter s',
      2),
     ('ythe american way forthat want a lighter sythe american way for people that want a lighter s',
      1),
     ('amazing new artra forthat want a lighter samazing new artra for people that want a lighter s',
      1),
     ('r complexion care forthat want a lighter sr complexion care for people that want a lighter s',
      1),
     ('a beautiful bride forthat want a lighter sa beautiful bride for people that want a lighter s',
      1),
     ('brides complexion forthat want a lighter sbrides complexion for people that want a lighter s',
      1),
     ('n becomes lighter forthat want a lighter sn becomes lighter for people that want a lighter s',
      1)]




```python
# Success is clearly associated with lighter skin
drum_text_object.concordance("successful", 120, 25)
```

    Displaying 25 of 884 matches:
    y makes you light protects skin against harsh sunlight successful people use ambi lighter clearer complexion with ambi a
    n give you the smooth clear lighter skin youve notices successful people have successful people use ambi scientifically 
    lear lighter skin youve notices successful people have successful people use ambi scientifically formulatedclear the ski
    emishes make it smoother and lighter within a few days successful people use ambi absolutely safeno burning or irritatio
     or irritation its even good for painful sunburnt skin successful people use ambi skin looked lighter and smoother prote
    and smoother protects your skin against harsh sunlight successful people use ambi worried by little pimples and spotskar
    by little pimples and spotskarroo takes these away too successful people use ambi keep my complexion looking light clear
    n give you the smooth clear lighter skin youve notices successful people have karroo morning karoo night makes you lovel
    n give you the smooth clear lighter skin youve notices successful people have karroo morning karoo night makes you lovel
    n give you the smooth clear lighter skin youve notices successful people have perfect skin beauty can be yours with the 
    n give you the smooth clear lighter skin youve notices successful people have hilite lightens and smooths skin scientifi
    n give you the smooth clear lighter skin youve notices successful people have brightens your skin fast and keeps it ligh
    n give you the smooth clear lighter skin youve notices successful people have karroo morning karoo night makes you lovel
    n give you the smooth clear lighter skin youve notices successful people have the two karroo creams keep my skin light c
    t clear light complexionyou can have 1 too with karroo successful people use ambi works wonders for skinlightens and smo
    te keeps your skin young and beautiful for much longer successful people use ambi starts lightening and smoothing your c
    ood and works so fast because it contains hydroquinone successful people use ambi helps clear blackheads pimples and fre
    e use ambi helps clear blackheads pimples and freckles successful people use ambi keep skin lighter smoother and softer 
    clearer it felt soft and smoothit was much lighter too successful people use ambi makes your skin and keep your skin loo
    ep your skin looking smoother lighter and soft as silk successful people use ambi lighter clearer complexion with ambi a
    n give you the smooth clear lighter skin youve notices successful people have successful people use ambi scientifically 
    lear lighter skin youve notices successful people have successful people use ambi scientifically formulatedclear the ski
    emishes make it smoother and lighter within a few days successful people use ambi absolutely safeno burning or irritatio
     or irritation its even good for painful sunburnt skin successful people use ambi clear light complexionyou can have 1 t
    n give you the smooth clear lighter skin youve notices successful people have use karroo creams and you too can be a bea



```python
# Most common words in context for with keyword 'successful'
drum_concord_list_obj = drum_text_object.concordance_list("successful", 50)
drum_concord_list = []
for con in drum_concord_list_obj:
    sent = "".join(list(con[4:]))
    drum_concord_list.append(sent)
drum_concord_list_freqs = nltk.FreqDist(drum_concord_list)
drum_concord_list_freqs.most_common(10)
```




    [(' skin youve noticespeople have karroo  skin youve notices successful people have karroo ',
      3),
     (' skin youve noticespeople have success skin youve notices successful people have success',
      2),
     ('cessful people havepeople use ambi scicessful people have successful people use ambi sci',
      2),
     ('r within a few dayspeople use ambi absr within a few days successful people use ambi abs',
      2),
     ('inst harsh sunlightpeople use ambi liginst harsh sunlight successful people use ambi lig',
      1),
     ('inful sunburnt skinpeople use ambi skiinful sunburnt skin successful people use ambi ski',
      1),
     ('inst harsh sunlightpeople use ambi worinst harsh sunlight successful people use ambi wor',
      1),
     ('akes these away toopeople use ambi keeakes these away too successful people use ambi kee',
      1),
     (' skin youve noticespeople have perfect skin youve notices successful people have perfect',
      1),
     (' skin youve noticespeople have hilite  skin youve notices successful people have hilite ',
      1)]




```python
drum_text_object.concordance("bride", 120, 25)
```

    Displaying 25 of 80 matches:
    ght complexion you too can look as lovely as a beautiful bride to make their skin lighter and lovelierlovelier and ligh
    ralightens and you too can look as lovely as a beautiful bride brightens skin lightens from the first day vanishes into
    ning your skin you too can look as lovely as a beautiful bride immediately keeps skin beautiful and clean makes it smoo
    oth and lovely you too can look as lovely as a beautiful bride mild and gentlekeeps skin free from blemishes and pimple
    es and pimples you too can look as lovely as a beautiful bride clears and lightens the skin smooths away blemishes and 
    ftens the skin you too can look as lovely as a beautiful bride gets to work immediately to give you a lighter lovelier 
    ier complexion you too can look as lovely as a beautiful bride a fair complexion clear skin that is blemish free and sp
    g this product you too can look as lovely as a beautiful bride your skin will grow lighter complexion smoother and blem
    s of your skin you too can look as lovely as a beautiful bride you will be more attractive more desirable will admire y
    ght complexion you too can look as lovely as a beautiful bride clears complexion in a few days any blemishes and spotsk
    ts rid of them you too can look as lovely as a beautiful bride the sooner you startexpect a lovely clear bright complex
    ght complexion you too can look as lovely as a beautiful bride to make their skin lighter and lovelierlovelier and ligh
    ralightens and you too can look as lovely as a beautiful bride brightens skin lightens from the first day vanishes into
    ning your skin you too can look as lovely as a beautiful bride immediately keeps skin beautiful and clean makes it smoo
    oth and lovely you too can look as lovely as a beautiful bride mild and gentlekeeps skin free from blemishes and pimple
    es and pimples you too can look as lovely as a beautiful bride ambi extramakes your skin positively lighter in is 4 day
    r and smoother you too can look as lovely as a beautiful bride for people that want a lighter smoother complexion makes
    nly a few days you too can look as lovely as a beautiful bride your skin will grow lighter complexion smoother and blem
    s of your skin you too can look as lovely as a beautiful bride you will be more attractive more desirable will admire y
    ear complexion you too can look as lovely as a beautiful bride your skin will grow lighter complexion smoother and blem
    s of your skin you too can look as lovely as a beautiful bride you will be more attractive more desirable will admire y
    ght complexion you too can look as lovely as a beautiful bride protects skin from the sun a light clear skin free from 
    es and pimples you too can look as lovely as a beautiful bride newest fastest acting skin lightening creamwill positive
    ear complexion you too can look as lovely as a beautiful bride lighter smoother skin in a few days cream can give you a
    ght complexion you too can look as lovely as a beautiful bride protects skin against harsh sunlight desire skin tone cr


<span id="drum_ngram" class="anchor"></span>

## Bigram and Trigrams in Drum


```python
# Collocations of commonly associated terms in Drum corpus, not necessarily bigrams
# Please see NLTK collocations documentation to understand how collocations are 
# selected: https://www.nltk.org/_modules/nltk/collocations.html

drum_text_object.collocations(num=100)
```

    hollywood seven; karroo morning; successful people; morning karoo;
    skin lightening; use ambi; beauty queen; karoo night; karroo creams;
    super rose; lighter smoother; people use; lightening cream; modern
    way; night makes; arms legs; cosmetic substance; miss south; americas
    great; lovely makes; legs neck; always wanted; look great; ugly spots;
    powerful name; lighter clearer; face arms; discolouring ugly; fast
    acting; german science; extra fast; one step; spots discolouring;
    removes ugly; quick acting; thoroughly removes; south africa; neck
    removes; acting medicinescleans; given thousands; velvety smooth;
    stronger super; double strength; light fantastic; lightening
    treatment; new cosmetic; cant lose; rose heman; says miss;
    complexionuse full; youll look; removes dark; quicker results; stops
    production; new stronger; aviva light; heman lotion; film star; top
    society; good looking; ambi youll; great looking; skin lightener;
    lighten dark; ugly blemishes; americans lighter; works fast; butter
    beautifies; burning rays; stays beautiful; queen complexionthey; dear
    heart; quick action; lightens face; dark spots; fragrant greaseless;
    lose says; white skin; morning karroo; smoother lovelier; makeup base;
    vanishing creamused; looking lovely; clearer complexionuse; acting
    ambi; clear natural; theyre seen; lightenerhas already; miss
    johannesburg; creams gave; clearer complexion; pimple freckle; skin
    whitens; lemon butter; skin thoroughly; natural look; youve noticed;
    skin leaving; lovely throughout; lighter brighter



```python
# Most common bigram frequencies
drum_bigram_list = list(nltk.bigrams(drum_text_object))
drum_bigram_freqs = nltk.FreqDist(drum_bigram_list)
pp.pprint(drum_bigram_freqs.most_common(75))
```

    [(('your', 'skin'), 2664),
     (('makes', 'you'), 1373),
     (('skin', 'lightening'), 1323),
     (('you', 'lovely'), 900),
     (('use', 'ambi'), 896),
     (('you', 'light'), 887),
     (('karroo', 'morning'), 881),
     (('hollywood', 'seven'), 807),
     (('karroo', 'creams'), 741),
     (('successful', 'people'), 684),
     (('a', 'lighter'), 674),
     (('lighter', 'smoother'), 634),
     (('beauty', 'queen'), 627),
     (('lightening', 'cream'), 621),
     (('beautiful', 'skin'), 620),
     (('night', 'makes'), 616),
     (('lovely', 'makes'), 616),
     (('spots', 'and'), 610),
     (('people', 'use'), 590),
     (('pimples', 'and'), 590),
     (('morning', 'karoo'), 564),
     (('karoo', 'night'), 564),
     (('can', 'have'), 563),
     (('for', 'a'), 555),
     (('in', 'the'), 535),
     (('keeps', 'you'), 530),
     (('smooth', 'and'), 513),
     (('the', 'best'), 500),
     (('and', 'pimples'), 484),
     (('extra', 'fast'), 480),
     (('look', 'great'), 476),
     (('while', 'it'), 465),
     (('keeps', 'skin'), 437),
     (('more', 'beautiful'), 426),
     (('light', 'skin'), 419),
     (('super', 'rose'), 419),
     (('to', 'lighten'), 411),
     (('lighter', 'clearer'), 404),
     (('and', 'blemishes'), 402),
     (('you', 'can'), 401),
     (('fast', 'acting'), 394),
     (('ugly', 'spots'), 393),
     (('it', 'lightens'), 388),
     (('the', 'beautiful'), 382),
     (('a', 'beauty'), 382),
     (('blemishes', 'and'), 374),
     (('smoother', 'and'), 372),
     (('lighten', 'your'), 372),
     (('beautiful', 'and'), 370),
     (('skin', 'the'), 368),
     (('new', 'skin'), 363),
     (('the', 'skin'), 359),
     (('skin', 'beautiful'), 356),
     (('and', 'light'), 352),
     (('the', 'modern'), 349),
     (('modern', 'way'), 349),
     (('light', 'fantastic'), 349),
     (('the', 'day'), 348),
     (('complexion', 'and'), 344),
     (('now', 'you'), 335),
     (('skin', 'lightener'), 333),
     (('your', 'complexion'), 327),
     (('lovelier', 'skin'), 324),
     (('morning', 'karroo'), 317),
     (('karroo', 'at'), 317),
     (('at', 'night'), 317),
     (('americas', 'great'), 314),
     (('lightening', 'treatment'), 314),
     (('makes', 'skin'), 312),
     (('and', 'lovely'), 311),
     (('miss', 'south'), 307),
     (('quick', 'acting'), 306),
     (('skin', 'you'), 303),
     (('skin', 'now'), 303),
     (('light', 'clear'), 300)]



```python
drum_bigram_freqs.plot(35, title="Drum Magazine Bigrams")
```


    
![png](output_38_0.png)
    





    <AxesSubplot:title={'center':'Drum Magazine Bigrams'}, xlabel='Samples', ylabel='Counts'>




```python
# Drum trigram frequencies
# Notice the most frequent trigrams relate to "makes you" construction. The sense that skin
# bleaching is about self-making and identity formation. There is a relationship between skin 
# colour and the American mythology of self-made success.
drum_trigram_list = list(nltk.trigrams(drum_text_object))
drum_trigram_freqs = nltk.FreqDist(drum_trigram_list)
pp.pprint(drum_trigram_freqs.most_common(75))
```

    [(('night', 'makes', 'you'), 616),
     (('makes', 'you', 'lovely'), 616),
     (('you', 'lovely', 'makes'), 616),
     (('lovely', 'makes', 'you'), 616),
     (('makes', 'you', 'light'), 616),
     (('karroo', 'morning', 'karoo'), 564),
     (('morning', 'karoo', 'night'), 564),
     (('skin', 'lightening', 'cream'), 544),
     (('successful', 'people', 'use'), 427),
     (('spots', 'and', 'pimples'), 425),
     (('people', 'use', 'ambi'), 393),
     (('while', 'it', 'lightens'), 385),
     (('ugly', 'spots', 'and'), 375),
     (('the', 'modern', 'way'), 349),
     (('lighten', 'your', 'skin'), 340),
     (('you', 'can', 'have'), 339),
     (('now', 'you', 'can'), 335),
     (('your', 'skin', 'the'), 330),
     (('karroo', 'morning', 'karroo'), 317),
     (('morning', 'karroo', 'at'), 317),
     (('karroo', 'at', 'night'), 317),
     (('at', 'night', 'makes'), 317),
     (('skin', 'lightening', 'treatment'), 314),
     (('pimples', 'and', 'blemishes'), 313),
     (('a', 'beauty', 'queen'), 301),
     (('for', 'a', 'lighter'), 300),
     (('karoo', 'night', 'makes'), 299),
     (('lovely', 'beauty', 'queen'), 289),
     (('fast', 'acting', 'ambi'), 288),
     (('a', 'lighter', 'smoother'), 287),
     (('skin', 'the', 'modern'), 285),
     (('skin', 'beautiful', 'and'), 283),
     (('a', 'few', 'days'), 282),
     (('lighter', 'clearer', 'complexion'), 274),
     (('your', 'skin', 'now'), 274),
     (('white', 'skin', 'whitens'), 273),
     (('skin', 'whitens', 'your'), 273),
     (('whitens', 'your', 'skin'), 273),
     (('skin', 'now', 'you'), 273),
     (('can', 'have', 'the'), 273),
     (('have', 'the', 'beautiful'), 273),
     (('the', 'beautiful', 'light'), 273),
     (('beautiful', 'light', 'skin'), 273),
     (('light', 'skin', 'you'), 273),
     (('skin', 'you', 'always'), 273),
     (('you', 'always', 'wanted'), 273),
     (('lightens', 'your', 'skin'), 270),
     (('for', 'great', 'looking'), 270),
     (('great', 'looking', 'skin'), 270),
     (('aviva', 'light', 'fantastic'), 266),
     (('karoo', 'night', 'keeps'), 265),
     (('night', 'keeps', 'you'), 265),
     (('keeps', 'you', 'lovely'), 265),
     (('you', 'lovely', 'keeps'), 265),
     (('lovely', 'keeps', 'you'), 265),
     (('keeps', 'you', 'light'), 265),
     (('keeps', 'skin', 'beautiful'), 264),
     (('in', 'the', 'world'), 258),
     (('name', 'in', 'skin'), 252),
     (('lightens', 'face', 'arms'), 247),
     (('face', 'arms', 'legs'), 247),
     (('arms', 'legs', 'neck'), 247),
     (('legs', 'neck', 'removes'), 247),
     (('neck', 'removes', 'dark'), 247),
     (('removes', 'dark', 'spots'), 247),
     (('dark', 'spots', 'discolouring'), 247),
     (('spots', 'discolouring', 'ugly'), 247),
     (('discolouring', 'ugly', 'blemishes'), 247),
     (('ugly', 'blemishes', 'and'), 247),
     (('blemishes', 'and', 'impurities'), 247),
     (('it', 'lightens', 'it'), 247),
     (('lightens', 'it', 'feeds'), 247),
     (('it', 'feeds', 'and'), 247),
     (('feeds', 'and', 'moisturizes'), 247),
     (('and', 'moisturizes', 'your'), 247)]



```python
# A plot of Drum Magazine trigrams
drum_trigram_freqs.plot(35, title="Drum Magazine Trigrams")
```


    
![png](output_40_0.png)
    





    <AxesSubplot:title={'center':'Drum Magazine Trigrams'}, xlabel='Samples', ylabel='Counts'>




```python
# While the quadgrams continue with the "makes you" construction, later formations introduce
# the logic of possibility in the present. The "now you can" phrasing figures prominently in ads
# which highlights the urgency of attaining associated attributes like whiteness, beauty,
# modernity, and success. 
drum_quadgram_list = list(nltk.ngrams(drum_text_object, 4))
drum_quadgram_freqs = nltk.FreqDist(drum_quadgram_list)
pp.pprint(drum_quadgram_freqs.most_common(75))
```

    [(('night', 'makes', 'you', 'lovely'), 616),
     (('makes', 'you', 'lovely', 'makes'), 616),
     (('you', 'lovely', 'makes', 'you'), 616),
     (('lovely', 'makes', 'you', 'light'), 616),
     (('karroo', 'morning', 'karoo', 'night'), 564),
     (('successful', 'people', 'use', 'ambi'), 393),
     (('ugly', 'spots', 'and', 'pimples'), 323),
     (('karroo', 'morning', 'karroo', 'at'), 317),
     (('morning', 'karroo', 'at', 'night'), 317),
     (('karroo', 'at', 'night', 'makes'), 317),
     (('at', 'night', 'makes', 'you'), 317),
     (('morning', 'karoo', 'night', 'makes'), 299),
     (('karoo', 'night', 'makes', 'you'), 299),
     (('your', 'skin', 'the', 'modern'), 285),
     (('skin', 'the', 'modern', 'way'), 285),
     (('white', 'skin', 'whitens', 'your'), 273),
     (('skin', 'whitens', 'your', 'skin'), 273),
     (('whitens', 'your', 'skin', 'now'), 273),
     (('your', 'skin', 'now', 'you'), 273),
     (('skin', 'now', 'you', 'can'), 273),
     (('now', 'you', 'can', 'have'), 273),
     (('you', 'can', 'have', 'the'), 273),
     (('can', 'have', 'the', 'beautiful'), 273),
     (('have', 'the', 'beautiful', 'light'), 273),
     (('the', 'beautiful', 'light', 'skin'), 273),
     (('beautiful', 'light', 'skin', 'you'), 273),
     (('light', 'skin', 'you', 'always'), 273),
     (('skin', 'you', 'always', 'wanted'), 273),
     (('for', 'great', 'looking', 'skin'), 270),
     (('morning', 'karoo', 'night', 'keeps'), 265),
     (('karoo', 'night', 'keeps', 'you'), 265),
     (('night', 'keeps', 'you', 'lovely'), 265),
     (('keeps', 'you', 'lovely', 'keeps'), 265),
     (('you', 'lovely', 'keeps', 'you'), 265),
     (('lovely', 'keeps', 'you', 'light'), 265),
     (('keeps', 'skin', 'beautiful', 'and'), 264),
     (('lightens', 'face', 'arms', 'legs'), 247),
     (('face', 'arms', 'legs', 'neck'), 247),
     (('arms', 'legs', 'neck', 'removes'), 247),
     (('legs', 'neck', 'removes', 'dark'), 247),
     (('neck', 'removes', 'dark', 'spots'), 247),
     (('removes', 'dark', 'spots', 'discolouring'), 247),
     (('dark', 'spots', 'discolouring', 'ugly'), 247),
     (('spots', 'discolouring', 'ugly', 'blemishes'), 247),
     (('discolouring', 'ugly', 'blemishes', 'and'), 247),
     (('ugly', 'blemishes', 'and', 'impurities'), 247),
     (('while', 'it', 'lightens', 'it'), 247),
     (('it', 'lightens', 'it', 'feeds'), 247),
     (('lightens', 'it', 'feeds', 'and'), 247),
     (('it', 'feeds', 'and', 'moisturizes'), 247),
     (('feeds', 'and', 'moisturizes', 'your'), 247),
     (('and', 'moisturizes', 'your', 'skin'), 247),
     (('moisturizes', 'your', 'skin', 'leaving'), 247),
     (('your', 'skin', 'leaving', 'it'), 247),
     (('skin', 'leaving', 'it', 'velvety'), 247),
     (('leaving', 'it', 'velvety', 'smooth'), 247),
     (('skin', 'beautiful', 'and', 'healthy'), 240),
     (('to', 'lighten', 'dark', 'skin'), 239),
     (('the', 'most', 'powerful', 'name'), 237),
     (('most', 'powerful', 'name', 'in'), 237),
     (('powerful', 'name', 'in', 'skin'), 237),
     (('name', 'in', 'skin', 'lightening'), 237),
     (('lighter', 'smoother', 'lovelier', 'skin'), 229),
     (('look', 'great', 'use', 'ambi'), 224),
     (('great', 'use', 'ambi', 'youll'), 224),
     (('use', 'ambi', 'youll', 'look'), 224),
     (('ambi', 'youll', 'look', 'great'), 224),
     (('quick', 'acting', 'medicinescleans', 'skin'), 221),
     (('acting', 'medicinescleans', 'skin', 'thoroughly'), 221),
     (('medicinescleans', 'skin', 'thoroughly', 'removes'), 221),
     (('skin', 'thoroughly', 'removes', 'ugly'), 221),
     (('thoroughly', 'removes', 'ugly', 'spots'), 221),
     (('removes', 'ugly', 'spots', 'and'), 221),
     (('spots', 'and', 'pimples', 'keeps'), 221),
     (('and', 'pimples', 'keeps', 'skin'), 221)]



```python
# A plot of Drum Magazine quadgrams
drum_quadgram_freqs.plot(35, title="Drum Magazine Quadgrams")
```


    
![png](output_42_0.png)
    





    <AxesSubplot:title={'center':'Drum Magazine Quadgrams'}, xlabel='Samples', ylabel='Counts'>



<span id="ebony_mfw" class="anchor"></span>

## Ebony Magazine MFW 


```python
# Convert dataframe into a list of lists of sentences for Ebony
ebony_catch_phrase_and_claims = pd.merge(catch_phrase_ebony, claims_ebony, right_index=True, left_index=True).dropna()
ebony_claims_phrase_list = ebony_catch_phrase_and_claims.values.tolist()
ebony_claims_phrase_list[:10]
```




    [['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '" - wakes up dark, dull complexion! Conceals ugly blotches, blemishes while it bleaches. Guarantees lovelier, lighter skin."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"If your skin doesn\'t look actually lighter after using Mercolized Wax Cream for just one week, your money will be cheerfully refunded."; "You\'ll see amazing results almost at once - as Mercolized Wax Cream\'s speedy bleaching action lightens your complexion, fades dark blotches, spots, and freckles, brings excessive skin oiliness under control."; "...works under the skin surface to bring about these marvelous results."; "Used by beautiful women for over 40 years."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"And you, too, can have a glamorous complexion!"; "…see your skin get a lighter, brighter, softer look."; "Its bleaching action works effectively inside your skin. Modern science knows no faster way of lightening skin."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"Don\'t let dull, dark skin rob you of romance. Don\'t let oiliness, big pores, blackheads cheat you of charm."; "This remarkable medicated ingredient works deep down within the skin to brighten and lighten it…"; "Soon your skin feels smoother and softer, fresh and fascinating, glowing and glamorous."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"Yes in just 7 days be delighted how fast and easy this doctor\'s fomrula lightens, brightens, and  helps clear skin or money back"; "It lightens, brightens and clears skin fast and at the same time fades blemishes, freckles and off-color spots."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"This Egyptian inspired formula proves you can make your complexion lovelier and keep it that way. So, enjoy a new feeling of distinction and beauty, convinceyourself of the quick, delightful results. What\'s more, MABS is easy and safe to use, gentle to tender, sensitive skin."'],
     ['"Lighter, brighter skin is irresistable"',
      '" - wakes up dark, dull complexion! Conceals ugly blotches, blemishes while it bleaches. Guarantees lovelier, lighter skin."'],
     ['"Lighter, brighter skin is irresistable"',
      '"If your skin doesn\'t look actually lighter after using Mercolized Wax Cream for just one week, your money will be cheerfully refunded."; "You\'ll see amazing results almost at once - as Mercolized Wax Cream\'s speedy bleaching action lightens your complexion, fades dark blotches, spots, and freckles, brings excessive skin oiliness under control."; "...works under the skin surface to bring about these marvelous results."; "Used by beautiful women for over 40 years."'],
     ['"Lighter, brighter skin is irresistable"',
      '"And you, too, can have a glamorous complexion!"; "…see your skin get a lighter, brighter, softer look."; "Its bleaching action works effectively inside your skin. Modern science knows no faster way of lightening skin."'],
     ['"Lighter, brighter skin is irresistable"',
      '"Don\'t let dull, dark skin rob you of romance. Don\'t let oiliness, big pores, blackheads cheat you of charm."; "This remarkable medicated ingredient works deep down within the skin to brighten and lighten it…"; "Soon your skin feels smoother and softer, fresh and fascinating, glowing and glamorous."']]




```python
# Tokenize and preprocess lists of sentences
ebony_word_corpus_list = []
for ad in ebony_claims_phrase_list:
    sents = [ch for ch in " ".join(ad).lower() if ch not in string.punctuation+"…"]
    ebony_word_corpus_list.append("".join(sents).split())
print(ebony_word_corpus_list[:5])            
```

    [['mercolized', 'wax', 'cream', 'guarantees', 'lighter', 'looking', 'skin', 'in', 'just', '7', 'days', 'or', 'money', 'back', 'wakes', 'up', 'dark', 'dull', 'complexion', 'conceals', 'ugly', 'blotches', 'blemishes', 'while', 'it', 'bleaches', 'guarantees', 'lovelier', 'lighter', 'skin'], ['mercolized', 'wax', 'cream', 'guarantees', 'lighter', 'looking', 'skin', 'in', 'just', '7', 'days', 'or', 'money', 'back', 'if', 'your', 'skin', 'doesnt', 'look', 'actually', 'lighter', 'after', 'using', 'mercolized', 'wax', 'cream', 'for', 'just', 'one', 'week', 'your', 'money', 'will', 'be', 'cheerfully', 'refunded', 'youll', 'see', 'amazing', 'results', 'almost', 'at', 'once', 'as', 'mercolized', 'wax', 'creams', 'speedy', 'bleaching', 'action', 'lightens', 'your', 'complexion', 'fades', 'dark', 'blotches', 'spots', 'and', 'freckles', 'brings', 'excessive', 'skin', 'oiliness', 'under', 'control', 'works', 'under', 'the', 'skin', 'surface', 'to', 'bring', 'about', 'these', 'marvelous', 'results', 'used', 'by', 'beautiful', 'women', 'for', 'over', '40', 'years'], ['mercolized', 'wax', 'cream', 'guarantees', 'lighter', 'looking', 'skin', 'in', 'just', '7', 'days', 'or', 'money', 'back', 'and', 'you', 'too', 'can', 'have', 'a', 'glamorous', 'complexion', 'see', 'your', 'skin', 'get', 'a', 'lighter', 'brighter', 'softer', 'look', 'its', 'bleaching', 'action', 'works', 'effectively', 'inside', 'your', 'skin', 'modern', 'science', 'knows', 'no', 'faster', 'way', 'of', 'lightening', 'skin'], ['mercolized', 'wax', 'cream', 'guarantees', 'lighter', 'looking', 'skin', 'in', 'just', '7', 'days', 'or', 'money', 'back', 'dont', 'let', 'dull', 'dark', 'skin', 'rob', 'you', 'of', 'romance', 'dont', 'let', 'oiliness', 'big', 'pores', 'blackheads', 'cheat', 'you', 'of', 'charm', 'this', 'remarkable', 'medicated', 'ingredient', 'works', 'deep', 'down', 'within', 'the', 'skin', 'to', 'brighten', 'and', 'lighten', 'it', 'soon', 'your', 'skin', 'feels', 'smoother', 'and', 'softer', 'fresh', 'and', 'fascinating', 'glowing', 'and', 'glamorous'], ['mercolized', 'wax', 'cream', 'guarantees', 'lighter', 'looking', 'skin', 'in', 'just', '7', 'days', 'or', 'money', 'back', 'yes', 'in', 'just', '7', 'days', 'be', 'delighted', 'how', 'fast', 'and', 'easy', 'this', 'doctors', 'fomrula', 'lightens', 'brightens', 'and', 'helps', 'clear', 'skin', 'or', 'money', 'back', 'it', 'lightens', 'brightens', 'and', 'clears', 'skin', 'fast', 'and', 'at', 'the', 'same', 'time', 'fades', 'blemishes', 'freckles', 'and', 'offcolor', 'spots']]



```python
# Collect all words into single sentence
ebony_word_corpus = []
for sent in ebony_word_corpus_list:
    for word in sent:
        ebony_word_corpus.append(word)
```


```python
# Total words in corpus and preview Drum word list 
ebony_num = len(ebony_word_corpus)
print(f"There are {ebony_num:,} words in the Ebony Magazine corpus. Remember, Drum Magazine contained {drum_num:,} words.")
print(ebony_word_corpus[:10])
```

    There are 632,675 words in the Ebony Magazine corpus. Remember, Drum Magazine contained 211,112 words.
    ['mercolized', 'wax', 'cream', 'guarantees', 'lighter', 'looking', 'skin', 'in', 'just', '7']



```python
# Create an NLTK Text object
ebony_text_object = nltk.Text(ebony_word_corpus)
```


```python
# Generate a sample concordance, displaying 120 characters of 25 matches
ebony_text_object.concordance("good",120,25)
```

    Displaying 25 of 205 matches:
    ghten glorify skin or money back i am excited about a new good looks cream for the whole family with gentle laboratory t
     radiance of that artra look now i am excited about a new good looks cream for the whole family that velvetysoft radiant
    ient proven safe for normal skin i am excited about a new good looks cream for the whole family now at last a complexion
     thats lighter brighter lovelier i am excited about a new good looks cream for the whole family long for the radiant glo
    skin artra softens your skin too i am excited about a new good looks cream for the whole family no more messy oldfashion
    leaching ingredient hydroquinone i am excited about a new good looks cream for the whole family clothes do the most for 
    of no faster way to lighten skin i am excited about a new good looks cream for the whole family photo caption now one ac
    learer smooth and radiantly soft i am excited about a new good looks cream for the whole family a delightful white cream
    ightens a toodark weathered skin i am excited about a new good looks cream for the whole family if you want to be pretty
     will say you look years younger i am excited about a new good looks cream for the whole family breathless enchanting me
    e contains no ammoniated mercury i am excited about a new good looks cream for the whole family if your skin doesnt look
    eautiful women for over 40 years i am excited about a new good looks cream for the whole family none i am excited about 
     cream for the whole family none i am excited about a new good looks cream for the whole family if your skin doesnt look
    eautiful women for over 40 years i am excited about a new good looks cream for the whole family no man can resist the al
    no faster way of lightening skin i am excited about a new good looks cream for the whole family now a complexion cream f
     helps keep skin clear and clean i am excited about a new good looks cream for the whole family if you face has a shiny 
    eback that is thrilling to watch i am excited about a new good looks cream for the whole family a delightful cream that 
    s famous formula with 10 lanolin i am excited about a new good looks cream for the whole family amazing new bleach and g
    it perfectly serve as a powder base and well you know how good lemon is for the skin apex lemon skin creme also works wo
    it perfectly serve as a powder base and well you know how good lemon is for the skin apex lemon skin creme also works wo
    it perfectly serve as a powder base and well you know how good lemon is for the skin apex lemon skin creme also works wo
    it perfectly serve as a powder base and well you know how good lemon is for the skin apex lemon skin creme also works wo
    it perfectly serve as a powder base and well you know how good lemon is for the skin apex lemon skin creme also works wo
    it perfectly serve as a powder base and well you know how good lemon is for the skin apex lemon skin creme also works wo
    it perfectly serve as a powder base and well you know how good lemon is for the skin apex lemon skin creme also works wo



```python
# Most common words in context for with keyword 'good'
drum_concord_list_obj = drum_text_object.concordance_list("good", 50)
drum_concord_list = []
for con in drum_concord_list_obj:
    sent = "".join(list(con[4:]))
    drum_concord_list.append(sent)
drum_concord_list_freqs = nltk.FreqDist(drum_concord_list)
drum_concord_list_freqs.most_common(10)
```




    [('as soon as its used isand works so fast becaas soon as its used is good and works so fast beca',
      10),
     ('or irritation its evenfor painful sunburnt sor irritation its even good for painful sunburnt s',
      4),
     ('lear bright complexionthings happen to a prelear bright complexion good things happen to a pre',
      1),
     ('made artralightens andthings happen to a premade artralightens and good things happen to a pre',
      1),
     (' brightening your skinthings happen to a pre brightening your skin good things happen to a pre',
      1),
     ('s it smooth and lovelythings happen to a pres it smooth and lovely good things happen to a pre',
      1),
     (' blemishes and pimplesthings happen to a pre blemishes and pimples good things happen to a pre',
      1),
     ('spots softens the skinthings happen to a prespots softens the skin good things happen to a pre',
      1),
     ('er lovelier complexionthings happen to a preer lovelier complexion good things happen to a pre',
      1),
     ('ned using this productthings happen to a prened using this product good things happen to a pre',
      1)]




```python
# Most common words in Ebony skin bleaching ads
# Remove stop words
ebony_word_corpus_no_stops = []
for word in ebony_word_corpus:
    if word not in STOPS: #add extra stops above
        ebony_word_corpus_no_stops.append(word)
ebony_freqs = nltk.FreqDist(ebony_word_corpus_no_stops)
ebony_most_common_counts = ebony_freqs.most_common(75)
ebony_most_common = []
for t,c in ebony_most_common_counts:
    ebony_most_common.append(t)
pp.pprint(ebony_most_common_counts)
```

    [('skin', 26920),
     ('cream', 8765),
     ('complexion', 5046),
     ('lighter', 4923),
     ('new', 4350),
     ('beauty', 4060),
     ('nadinola', 3785),
     ('dark', 3515),
     ('artra', 3448),
     ('use', 3307),
     ('look', 3242),
     ('glow', 3139),
     ('spots', 2922),
     ('brighter', 2896),
     ('tone', 2557),
     ('blackheads', 2432),
     ('bleach', 2402),
     ('bleaching', 2395),
     ('face', 2375),
     ('see', 2321),
     ('helps', 2309),
     ('clear', 2255),
     ('lovelier', 2177),
     ('blemishes', 2135),
     ('works', 2080),
     ('contains', 1974),
     ('smooth', 1908),
     ('even', 1835),
     ('action', 1767),
     ('away', 1757),
     ('effective', 1744),
     ('treatment', 1733),
     ('ultra', 1673),
     ('ingredient', 1667),
     ('smoother', 1655),
     ('clearer', 1645),
     ('pores', 1642),
     ('women', 1590),
     ('fade', 1578),
     ('formula', 1508),
     ('try', 1501),
     ('beautiful', 1497),
     ('esoterica', 1492),
     ('care', 1491),
     ('ambi', 1475),
     ('one', 1461),
     ('soft', 1424),
     ('help', 1404),
     ('results', 1336),
     ('hydroquinone', 1321),
     ('day', 1321),
     ('surface', 1317),
     ('special', 1227),
     ('days', 1216),
     ('mercolized', 1198),
     ('using', 1191),
     ('dry', 1179),
     ('peelerpak', 1158),
     ('looking', 1154),
     ('areas', 1128),
     ('lightens', 1122),
     ('hands', 1118),
     ('fades', 1109),
     ('actually', 1107),
     ('palmers', 1085),
     ('age', 1079),
     ('blotches', 1078),
     ('white', 1065),
     ('makes', 1062),
     ('years', 1060),
     ('radiant', 1056),
     ('start', 1048),
     ('softer', 1036),
     ('makeup', 1027),
     ('comes', 1009)]



```python
# A plot of Ebony Magazine MFW
ebony_freqs.plot(35, title="Ebony Magazine MFW")
```


    
![png](output_52_0.png)
    





    <AxesSubplot:title={'center':'Ebony Magazine MFW'}, xlabel='Samples', ylabel='Counts'>




```python
# Skin bleaching products are associated with newness, like many consumer products
# The new "feeling" is about "discovery," almost as if skin bleaching is about excavating
# a new, lighter sense of self. 
ebony_text_object.concordance("new",120,50)
```

    Displaying 50 of 4350 matches:
     your complexion lovelier and keep it that way so enjoy a new feeling of distinction and beauty convinceyourself of the
     your complexion lovelier and keep it that way so enjoy a new feeling of distinction and beauty convinceyourself of the
     your complexion lovelier and keep it that way so enjoy a new feeling of distinction and beauty convinceyourself of the
     your complexion lovelier and keep it that way so enjoy a new feeling of distinction and beauty convinceyourself of the
     your complexion lovelier and keep it that way so enjoy a new feeling of distinction and beauty convinceyourself of the
     lighter lovelier skin thats both thorough and gentle its new artra skin tone cream with hydroquinone the miracleaction
    ep skin clear and clean biggest beauty value you ever saw new bleach and glow leaves your skin shades lighter clearer s
    rsened areas clears blemishes blackheads too now in handy new plastic squeeze tube fits perfectly into purse or makeup 
    auty value you ever saw see your dull dark skin take on a new lighter brighter softer smoother look its bleachign actio
    ghter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hydroquinone dev
    ghter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hydroquinone dev
     lighter lovelier skin thats both thorough and gentle its new artra skin tone cream with hydroquinone the miracleaction
    ghter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hydroquinone dev
    y tested proven effective by thousands of satisfied users new bleach and glow leaves your skin shades lighter clearer s
    rsened areas clears blemishes blackheads too now in handy new plastic squeeze tube fits perfectly into purse or makeup 
    ghter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hydroquinone dev
    ghter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hydroquinone dev
    ands of satisfied users see your dull dark skin take on a new lighter brighter softer smoother look its bleachign actio
     lighter lovelier skin thats both thorough and gentle its new artra skin tone cream with hydroquinone the miracleaction
    y yours with bleach and glow the look of total loveliness new bleach and glow leaves your skin shades lighter clearer s
    rsened areas clears blemishes blackheads too now in handy new plastic squeeze tube fits perfectly into purse or makeup 
    ook of total loveliness see your dull dark skin take on a new lighter brighter softer smoother look its bleachign actio
     lighter lovelier skin thats both thorough and gentle its new artra skin tone cream with hydroquinone the miracleaction
    rantees lighter looking skin in just 7 days or money back new bleach and glow leaves your skin shades lighter clearer s
    rsened areas clears blemishes blackheads too now in handy new plastic squeeze tube fits perfectly into purse or makeup 
    st 7 days or money back see your dull dark skin take on a new lighter brighter softer smoother look its bleachign actio
     lighter lovelier skin thats both thorough and gentle its new artra skin tone cream with hydroquinone the miracleaction
    clean the prettiest valentines have lighter brighter skin new bleach and glow leaves your skin shades lighter clearer s
    rsened areas clears blemishes blackheads too now in handy new plastic squeeze tube fits perfectly into purse or makeup 
    e lighter brighter skin see your dull dark skin take on a new lighter brighter softer smoother look its bleachign actio
    hat below the surface of the skin is a natural base for a new light creamy complexion no matter how drab your skin is n
    n is nowno matter what youve tried in the past an amazing new discovery unveils the light creamy loveliness the true sk
    ed complexion are now casting it offstepping into a great new day with a glowing glamorous radiant complexion it can ha
    hat below the surface of the skin is a natural base for a new light creamy complexion no matter how drab your skin is n
    n is nowno matter what youve tried in the past an amazing new discovery unveils the light creamy loveliness the true sk
    ed complexion are now casting it offstepping into a great new day with a glowing glamorous radiant complexion it can ha
    hat below the surface of the skin is a natural base for a new light creamy complexion no matter how drab your skin is n
    n is nowno matter what youve tried in the past an amazing new discovery unveils the light creamy loveliness the true sk
    ed complexion are now casting it offstepping into a great new day with a glowing glamorous radiant complexion it can ha
     as bright as mine unlock the glorious hidden secret of a new lighter creamier complexion which lies waiting under your
     as bright as mine unlock the glorious hidden secret of a new lighter creamier complexion which lies waiting under your
     as bright as mine unlock the glorious hidden secret of a new lighter creamier complexion which lies waiting under your
     as bright as mine unlock the glorious hidden secret of a new lighter creamier complexion which lies waiting under your
    hat below the surface of the skin is a natural base for a new light creamy complexion no matter how drab your skin is n
    n is nowno matter what youve tried in the past an amazing new discovery unveils the light creamy loveliness the true sk
    ed complexion are now casting it offstepping into a great new day with a glowing glamorous radiant complexion it can ha
    ant glow that says here is a woman who cares for her skin new artra skin tone cream was developed in modern laboratorie
    ra skin tone cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra light
     copy the dramatic and almost unbelievable results from a new and different kind of skin cream almost like magic brown 
    m breathless enchanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter



```python
# Most common words in context for with keyword 'new'
ebony_concord_list_obj = ebony_text_object.concordance_list("new", 50)
ebony_concord_list = []
for con in ebony_concord_list_obj:
    sent = "".join(list(con[4:]))
    ebony_concord_list.append(sent)
ebony_concord_list_freqs = nltk.FreqDist(ebony_concord_list)
ebony_concord_list_freqs.most_common(10)
```




    [('it that way so enjoy afeeling of distinctionit that way so enjoy a new feeling of distinction',
      5),
     ('nder of modern scienceartra skin tone cream nder of modern science new artra skin tone cream ',
      5),
     ('horough and gentle itsartra skin tone cream horough and gentle its new artra skin tone cream ',
      4),
     ('heads too now in handyplastic squeeze tube fheads too now in handy new plastic squeeze tube f',
      4),
     ('ll dark skin take on alighter brighter softell dark skin take on a new lighter brighter softe',
      3),
     ('uty value you ever sawbleach and glow leavesuty value you ever saw new bleach and glow leaves',
      1),
     ('nds of satisfied usersbleach and glow leavesnds of satisfied users new bleach and glow leaves',
      1),
     ('ok of total lovelinessbleach and glow leavesok of total loveliness new bleach and glow leaves',
      1),
     ('t 7 days or money backbleach and glow leavest 7 days or money back new bleach and glow leaves',
      1)]




```python
# The discovery of new products and formulas is aligned with the discovery of beauty, 
# and even a community of other men and women using these products. 
ebony_text_object.concordance("discover",120,230)
```

    Displaying 230 of 230 matches:
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    he lightest loveliest exciting skin of your life try it discover for yourself how soft light and radiant your skin can b
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    ur looks use greaseless nadinola deluxe bleaching cream discover how really lovely your skin can be simply give yourself
    unds those horrid age spotsfade them out with esoterica discover how really lovely your skin can be simply give yourself
    bestmercolized cream guarantees glamorously fairer skin discover how really lovely your skin can be simply give yourself
    mpounds lighter clearer skin too may be yours in 7 days discover how really lovely your skin can be simply give yourself
     times everyone knows you care enough to use the finest discover how really lovely your skin can be simply give yourself
    by far dependable quality skin care products since 1840 discover how really lovely your skin can be simply give yourself
    hter softer with a new youthful glow try skintona today discover how pretty you really are whats more fun than being a g
    hter softer with a new youthful glow try skintona today discover how pretty you really are bleach and glow cream makes y
    hter softer with a new youthful glow try skintona today discover how pretty you really are those horrid age spotsfade th
    hter softer with a new youthful glow try skintona today discover how pretty you really are lighter clearer skin too may 
    hter softer with a new youthful glow try skintona today discover how pretty you really are darling ive never seen your s
    hter softer with a new youthful glow try skintona today discover how pretty you really are for those who can afford the 
    hter softer with a new youthful glow try skintona today discover how pretty you really are for those who can afford the 
    hter softer with a new youthful glow try skintona today discover how pretty you really are this may be the most exciting
    hter softer with a new youthful glow try skintona today discover how pretty you really are use persulan proved more than
    hter softer with a new youthful glow try skintona today discover how pretty you really are lighter clearer skin too may 
    hter softer with a new youthful glow try skintona today discover how pretty you really are those horrid age spotsfade th
    hter softer with a new youthful glow try skintona today discover how pretty you really are darling your skin is so thril
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    o get todays best buy in beauty at a sensational saving discover a lighter lovelier complexion as thousands of women alr
    artra skin tone cream at your favorite cosmetic counter discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    enetration action goes deep feel and see the difference discover ultra nadinola for a brighter lighter more eventoned lo
    eams today start to bleach and glow your drab skin away discover ultra nadinola for a brighter lighter more eventoned lo
    ightens darker pigment areas to a brigher lovelier tone discover ultra nadinola for a brighter lighter more eventoned lo
    your skin glows with new brighter naturallooking beauty discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    art using black and white bleaching cream this very day discover ultra nadinola for a brighter lighter more eventoned lo
    eas to produce a brighter more eventoned glowing effect discover ultra nadinola for a brighter lighter more eventoned lo
    n to fade ugly brown spots and to lighten darkened neck discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    kin whitener for lighter brighter smoother looking skin discover ultra nadinola for a brighter lighter more eventoned lo
    xion stage a beauty comeback that is thrilling to watch discover ultra nadinola for a brighter lighter more eventoned lo
    eams today start to bleach and glow your drab skin away discover ultra nadinola for a brighter lighter more eventoned lo
    artra skin tone cream at your favorite cosmetic counter discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    enetration action goes deep feel and see the difference discover ultra nadinola for a brighter lighter more eventoned lo
    eams today start to bleach and glow your drab skin away discover ultra nadinola for a brighter lighter more eventoned lo
    ightens darker pigment areas to a brigher lovelier tone discover ultra nadinola for a brighter lighter more eventoned lo
    your skin glows with new brighter naturallooking beauty discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    art using black and white bleaching cream this very day discover ultra nadinola for a brighter lighter more eventoned lo
    eas to produce a brighter more eventoned glowing effect discover ultra nadinola for a brighter lighter more eventoned lo
    n to fade ugly brown spots and to lighten darkened neck discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    kin whitener for lighter brighter smoother looking skin discover ultra nadinola for a brighter lighter more eventoned lo
    xion stage a beauty comeback that is thrilling to watch discover ultra nadinola for a brighter lighter more eventoned lo
    eams today start to bleach and glow your drab skin away discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
     deluxe and see your complexion stage a beauty comeback discover ultra nadinola for a brighter lighter more eventoned lo
    ightens darker pigment areas to a brigher lovelier tone discover ultra nadinola for a brighter lighter more eventoned lo
    art using black and white bleaching cream this very day discover ultra nadinola for a brighter lighter more eventoned lo
    your skin glows with new brighter naturallooking beauty discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    eams today start to bleach and glow your drab skin away discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    ightens darker pigment areas to a brigher lovelier tone discover ultra nadinola for a brighter lighter more eventoned lo
    n to fade ugly brown spots and to lighten darkened neck discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    artra skin tone cream at your favorite cosmetic counter discover ultra nadinola for a brighter lighter more eventoned lo
    kin whitener for lighter brighter smoother looking skin discover ultra nadinola for a brighter lighter more eventoned lo
    e radiant and you will be too new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    more eventoned glowing effect new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    ck that is thrilling to watch new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    g begin using esoterica today new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    bleaching cream this very day new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    righter naturallooking beauty new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    ilized hydroquinone compounds new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    eas to produce a brighter more eventoned glowing effect discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    artra skin tone cream at your favorite cosmetic counter discover ultra nadinola for a brighter lighter more eventoned lo
    kin whitener for lighter brighter smoother looking skin discover ultra nadinola for a brighter lighter more eventoned lo
     deluxe and see your complexion stage a beauty comeback discover ultra nadinola for a brighter lighter more eventoned lo
    ightens darker pigment areas to a brigher lovelier tone discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
     deluxe and see your complexion stage a beauty comeback discover ultra nadinola for a brighter lighter more eventoned lo
    eams today start to bleach and glow your drab skin away discover ultra nadinola for a brighter lighter more eventoned lo
    kin whitener for lighter brighter smoother looking skin discover ultra nadinola for a brighter lighter more eventoned lo
    ightens darker pigment areas to a brigher lovelier tone discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    your skin glows with new brighter naturallooking beauty discover ultra nadinola for a brighter lighter more eventoned lo
    eams today start to bleach and glow your drab skin away discover ultra nadinola for a brighter lighter more eventoned lo
    art using black and white bleaching cream this very day discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
     deluxe and see your complexion stage a beauty comeback discover ultra nadinola for a brighter lighter more eventoned lo
    ies inc trademark for stabilized hydroquinone compounds discover ultra nadinola for a brighter lighter more eventoned lo
    art using black and white bleaching cream this very day discover ultra nadinola for a brighter lighter more eventoned lo
    ver try ityour skin will be radiant and you will be too discover ultra nadinola for a brighter lighter more eventoned lo
    shes fairer younger looking begin using esoterica today discover ultra nadinola for a brighter lighter more eventoned lo
    ightens darker pigment areas to a brigher lovelier tone discover ultra nadinola for a brighter lighter more eventoned lo
    exion stage a beauty comeback new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    more eventoned glowing effect new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    e radiant and you will be too new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    bleaching cream this very day new fasterworking formula discover ultra nadinola for a brighter lighter more eventoned lo
    pecial ingredient am in deluxe and regular formulations discover how really lovely your skin can be simply give yourself
    so many clear and brighten their complexions with artra discover how really lovely your skin can be simply give yourself
    nce that comes with clear radiant skin natural beauties discover how really lovely your skin can be simply give yourself
    ight side of life a whole new world has opened up to me discover how really lovely your skin can be simply give yourself
    with clear radiant skin soften dry skin while you sleep discover how really lovely your skin can be simply give yourself
     skin try ityou will love your new look and he will too discover how really lovely your skin can be simply give yourself
    lear radiant skin lighter brighter skin is irresistable discover how really lovely your skin can be simply give yourself
    eam brighten your skin brighten your life with nadinola discover how really lovely your skin can be simply give yourself
    brings a lighter brighter look to women round the world discover how really lovely your skin can be simply give yourself
    skin those horrid age spotsfade them out with esoterica discover how really lovely your skin can be simply give yourself
    th clear radiant skin share my secret for lovelier skin discover how really lovely your skin can be simply give yourself
     skin try ityou will love your new look and he will too discover how really lovely your skin can be simply give yourself
    lear radiant skin lighter brighter skin is irresistable discover how really lovely your skin can be simply give yourself
    nce that comes with clear radiant skin natural beauties discover how really lovely your skin can be simply give yourself
    ight side of life a whole new world has opened up to me discover how really lovely your skin can be simply give yourself
     skin try ityou will love your new look and he will too discover how really lovely your skin can be simply give yourself
     irresistable depend on black and white for skin beauty discover how really lovely your skin can be simply give yourself
    cess cream for fairer clearer naturallooking loveliness discover how really lovely your skin can be simply give yourself
    ve one cream could do all that at first we didnt either discover how really lovely your skin can be simply give yourself
     perspiration odorsmakes you sure youre nice to be near discover the hidden beauty that is naturally you the proof came 
     for normal skin cant stain clothes because it vanishes discover the hidden beauty that is naturally you underneath your
    se it faithfully and smile at the dreamier creamier you discover the hidden beauty that is naturally you the flowerand t
    enjoy the confidence that comes with clear radiant skin discover the hidden beauty that is naturally you join in the fun
    ly effective by women of all color tones the world over discover the hidden beauty that is naturally you discover how re
    d over discover the hidden beauty that is naturally you discover how really lovely your skin can be simply give yourself
    omen of all color tones the world over natural beauties discover how really lovely your skin can be simply give yourself
    th a lighter clearer skin try itfor an exciting new you discover how really lovely your skin can be simply give yourself
    cess cream for fairer clearer naturallooking loveliness discover how really lovely your skin can be simply give yourself
     developed it for lovelier skin try new esoterica today discover the hidden beauty that is naturally you brighten up you
    d over try ityou will love your new look and he will to discover the hidden beauty that is naturally you underneath your
    se it faithfully and smile at the dreamier creamier you discover the hidden beauty that is naturally you weathered brown
     your skin for fairer clearer naturallooking loveliness discover how really lovely your skin can be fragrant creamy whit
     irresistable depend on black and white for skin beauty discover how really lovely your skin can be fragrant creamy whit
    inola bring out the hidden beauty that is naturally you discover the hidden beauty that is naturally you discover how re
    ly you discover the hidden beauty that is naturally you discover how really lovely your skin can be fragrant creamy whit
     perspiration odorsmakes you sure youre nice to be near discover the hidden beauty that is naturally you and you too can
    art using black and white bleaching cream this very day discover the hidden beauty that is naturally you underneath the 
    adiant brightness that blooms in a bleach and glow skin discover how really lovely your skin can be fragrant creamy whit
     your skin for fairer clearer naturallooking loveliness discover how really lovely your skin can be fragrant creamy whit
    iant glow of love and romance our annual thank you sale discover how really lovely your skin can be fragrant creamy whit
    th a lighter clearer skin try itfor an exciting new you discover how really lovely your skin can be fragrant creamy whit
    ance those horrid age spotsfade them out with esoterica discover how really lovely your skin can be fragrant creamy whit
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el
    ven toned and flawless without a trace of discoloration discover why women and men of all ages are using dermacure to el



```python
# Most common words in context for with keyword 'discover'. In this case, there is only one
# slogan that repeats in the corpus, but it repeats 25 times so it appears in many ads.
ebony_concord_list_obj = ebony_text_object.concordance_list("discover", 70)
ebony_concord_list = []
for con in ebony_concord_list_obj:
    sent = "".join(list(con[4:]))
    ebony_concord_list.append(sent)
ebony_concord_list_freqs = nltk.FreqDist(ebony_concord_list)
ebony_concord_list_freqs.most_common(10)
```




    [('iting skin of your life try itfor yourself how soft light aniting skin of your life try it discover for yourself how soft light an',
      25)]



<span id="ebony_ngram" class="anchor"></span>

## Bigram and Trigrams in Ebony


```python
# Collocations of commonly associated terms in Ebony corpus, not necessarily bigrams
# Please see NLTK collocations documentation to understand how collocations are 
# selected: https://www.nltk.org/_modules/nltk/collocations.html

ebony_text_object.collocations(num=100)
```

    lighter brighter; mercolized wax; bleaching cream; skin tone; fred
    palmers; money back; enlarged pores; ultra nadinola; ammoniated
    mercury; dark spots; tone cream; nadinola deluxe; modern science;
    basenot greasyvanishes; greasyvanishes immediatelyfor; externally
    caused; laboratories inc; inc trademark; drake laboratories; dont let;
    million times; powder basenot; smootha powder; peeling treatment;
    times everyone; white bleaching; persulan proved; wax cream;
    repeatedly prescribed; start using; everyone knows; skin whitener;
    makeup base; bleaching action; exclusive formula; palmers skin;
    ingredients repeatedly; refining enlarged; ponds vanishing;
    hydroquinone compounds; stabilized hydroquinone; brown spots; eye
    shadows; dry blotchy; young people; helps keep; razor bumps; gplus
    action; persu persu; care enough; vanishing cream; large pores;
    cheerfully refunded; bright nadinolalight; peelerpak treatment; dull
    dark; medicated ingredient; use persulan; highly recommended;
    consecutive nights; science knows; daily routine; faster way; age
    lines; ingredient persu; wonderful ingredient; quickie smearon;
    smearon method; youll see; effective ingredients; age spots;
    superficial quickie; glow cream; acne large; skin success; brings
    excessive; creams speedy; removing blackheads; recommended aid; one
    week; works effectively; using black; acne scars; suddenly comes; men
    women; action works; useit contains; alive easy; helen williams; fast
    acting; lighter clearer; comes alive; new wonderful; nothing else;
    effectively inside; face arms; keep skin; face neck; deluxe bleaching;
    speedy bleaching



```python
# Most common bigram frequencies in Ebony Magazine. It is notable that Dr. Fred Palmer's
# skin cream is still manufactured in the US: 
# https://summitlabsinc.com/product/dr-fred-summit-skin-whitener-tone-and-bleach-cream/
ebony_bigram_list = list(nltk.bigrams(ebony_text_object))
ebony_bigram_freqs = nltk.FreqDist(ebony_bigram_list)
pp.pprint(ebony_bigram_freqs.most_common(75))
```

    [(('your', 'skin'), 6006),
     (('lighter', 'brighter'), 1916),
     (('skin', 'tone'), 1881),
     (('bleach', 'and'), 1735),
     (('and', 'glow'), 1616),
     (('bleaching', 'cream'), 1416),
     (('a', 'lighter'), 1210),
     (('tone', 'cream'), 1185),
     (('your', 'complexion'), 1160),
     (('the', 'skin'), 1123),
     (('blackheads', 'and'), 1034),
     (('dark', 'spots'), 1012),
     (('mercolized', 'wax'), 996),
     (('will', 'be'), 996),
     (('of', 'your'), 962),
     (('palmers', 'skin'), 941),
     (('your', 'face'), 932),
     (('spots', 'and'), 919),
     (('ultra', 'nadinola'), 914),
     (('as', 'a'), 859),
     (('skin', 'whitener'), 853),
     (('artra', 'skin'), 825),
     (('for', 'a'), 810),
     (('to', 'use'), 798),
     (('glow', 'cream'), 780),
     (('black', 'and'), 775),
     (('and', 'white'), 775),
     (('wax', 'cream'), 771),
     (('in', 'the'), 771),
     (('skin', 'to'), 764),
     (('skin', 'beauty'), 761),
     (('nadinola', 'deluxe'), 751),
     (('can', 'be'), 741),
     (('as', 'it'), 722),
     (('brighter', 'skin'), 721),
     (('dr', 'fred'), 689),
     (('on', 'the'), 681),
     (('in', 'just'), 680),
     (('money', 'back'), 680),
     (('over', '40'), 674),
     (('40', 'years'), 674),
     (('fred', 'palmers'), 665),
     (('skin', 'success'), 664),
     (('to', 'help'), 663),
     (('bleaching', 'action'), 661),
     (('on', 'your'), 661),
     (('lovelier', 'skin'), 653),
     (('skin', 'will'), 652),
     (('skin', 'in'), 650),
     (('by', 'doctors'), 637),
     (('the', 'new'), 634),
     (('the', 'world'), 634),
     (('for', 'skin'), 633),
     (('and', 'see'), 624),
     (('white', 'bleaching'), 623),
     (('is', 'the'), 620),
     (('contains', 'the'), 612),
     (('for', 'you'), 611),
     (('soft', 'and'), 611),
     (('enlarged', 'pores'), 611),
     (('skin', 'care'), 602),
     (('keep', 'skin'), 600),
     (('your', 'money'), 596),
     (('use', 'it'), 594),
     (('7', 'days'), 589),
     (('see', 'your'), 579),
     (('a', 'new'), 578),
     (('with', 'the'), 572),
     (('of', 'the'), 567),
     (('skin', 'cream'), 554),
     (('more', 'than'), 549),
     (('peeling', 'treatment'), 540),
     (('for', 'over'), 536),
     (('and', 'you'), 534),
     (('for', 'normal'), 533)]



```python
# A plot of Ebony bigrams
ebony_bigram_freqs.plot(35, title="Ebony Magazine Bigrams")
```


    
![png](output_60_0.png)
    





    <AxesSubplot:title={'center':'Ebony Magazine Bigrams'}, xlabel='Samples', ylabel='Counts'>




```python
# Ebony trigram frequencies exposes a distinct phrasing from Drum ads, with assurances about 
# effectiveness and provenness of the product. Perhaps there is more skepticism to use of these
# products and the need to assure that 20 million people have used these is products might
# give confidence to otherwise suspicious products. The presence of the "normal skin" construction
# is perhaps the most leading phrasing here. 
ebony_trigram_list = list(nltk.trigrams(ebony_text_object))
ebony_trigram_freqs = nltk.FreqDist(ebony_trigram_list)
pp.pprint(ebony_trigram_freqs.most_common(75))
```

    [(('bleach', 'and', 'glow'), 1583),
     (('skin', 'tone', 'cream'), 1118),
     (('black', 'and', 'white'), 775),
     (('mercolized', 'wax', 'cream'), 771),
     (('and', 'glow', 'cream'), 741),
     (('artra', 'skin', 'tone'), 726),
     (('over', '40', 'years'), 674),
     (('dr', 'fred', 'palmers'), 665),
     (('lighter', 'brighter', 'skin'), 645),
     (('and', 'white', 'bleaching'), 623),
     (('white', 'bleaching', 'cream'), 610),
     (('palmers', 'skin', 'whitener'), 573),
     (('fred', 'palmers', 'skin'), 552),
     (('for', 'over', '40'), 536),
     (('a', 'lighter', 'brighter'), 516),
     (('contains', 'the', 'new'), 511),
     (('to', 'use', 'the'), 478),
     (('helps', 'keep', 'skin'), 473),
     (('nadinola', 'bleaching', 'cream'), 461),
     (('and', 'see', 'your'), 425),
     (('by', 'doctors', 'for'), 422),
     (('bleaching', 'cream', 'as'), 421),
     (('cream', 'as', 'directed'), 421),
     (('for', 'normal', 'skin'), 421),
     (('skin', 'soft', 'and'), 414),
     (('bleaching', 'action', 'works'), 411),
     (('dry', 'blotchy', 'skin'), 400),
     (('the', 'world', 'over'), 390),
     (('on', 'your', 'face'), 386),
     (('easy', 'and', 'pleasant'), 382),
     (('and', 'pleasant', 'to'), 382),
     (('blackheads', 'and', 'refining'), 382),
     (('keep', 'skin', 'soft'), 381),
     (('for', 'skin', 'care'), 380),
     (('you', 'will', 'be'), 376),
     (('skin', 'can', 'be'), 375),
     (('use', 'persulan', 'proved'), 375),
     (('persulan', 'proved', 'more'), 375),
     (('proved', 'more', 'than'), 375),
     (('more', 'than', '20'), 375),
     (('than', '20', 'million'), 375),
     (('20', 'million', 'times'), 375),
     (('million', 'times', 'everyone'), 375),
     (('times', 'everyone', 'knows'), 375),
     (('everyone', 'knows', 'you'), 375),
     (('knows', 'you', 'care'), 375),
     (('you', 'care', 'enough'), 375),
     (('care', 'enough', 'to'), 375),
     (('enough', 'to', 'use'), 375),
     (('use', 'the', 'finest'), 375),
     (('prescribed', 'by', 'doctors'), 374),
     (('soft', 'and', 'smootha'), 373),
     (('and', 'smootha', 'powder'), 373),
     (('smootha', 'powder', 'basenot'), 373),
     (('powder', 'basenot', 'greasyvanishes'), 373),
     (('basenot', 'greasyvanishes', 'immediatelyfor'), 373),
     (('greasyvanishes', 'immediatelyfor', 'skin'), 373),
     (('immediatelyfor', 'skin', 'blemishes'), 373),
     (('skin', 'blemishes', 'contains'), 373),
     (('blemishes', 'contains', 'the'), 373),
     (('the', 'new', 'wonderful'), 373),
     (('new', 'wonderful', 'ingredient'), 373),
     (('wonderful', 'ingredient', 'persu'), 373),
     (('ingredient', 'persu', 'persu'), 373),
     (('persu', 'persu', 'is'), 373),
     (('persu', 'is', 'drake'), 373),
     (('is', 'drake', 'laboratories'), 373),
     (('drake', 'laboratories', 'inc'), 373),
     (('laboratories', 'inc', 'trademark'), 373),
     (('inc', 'trademark', 'for'), 373),
     (('trademark', 'for', 'stabilized'), 373),
     (('for', 'stabilized', 'hydroquinone'), 373),
     (('stabilized', 'hydroquinone', 'compounds'), 373),
     (('your', 'money', 'back'), 371),
     (('palmers', 'skin', 'success'), 360)]



```python
# A plot of trigrams in Ebony Magazine
ebony_trigram_freqs.plot(35, title="Ebony Magazine Trigrams")
```


    
![png](output_62_0.png)
    





    <AxesSubplot:title={'center':'Ebony Magazine Trigrams'}, xlabel='Samples', ylabel='Counts'>




```python
# Quaddgrams for Ebony Magazine emphasize the "finest" skin and "ingredients." The presence of 
# scientific words of assurance like "laboratories," "doctors," and "prescribed" suggests these
# ads are seeking scientific legitimacy. The most explicit alignment of these ideas coming together
# in the ('your', 'skin', 'modern', 'science') quadgram.
ebony_quadgram_list = list(nltk.ngrams(ebony_text_object, 4))
ebony_quadgram_freqs = nltk.FreqDist(ebony_quadgram_list)
pp.pprint(ebony_quadgram_freqs.most_common(75))
```

    [(('artra', 'skin', 'tone', 'cream'), 726),
     (('bleach', 'and', 'glow', 'cream'), 710),
     (('black', 'and', 'white', 'bleaching'), 623),
     (('and', 'white', 'bleaching', 'cream'), 610),
     (('dr', 'fred', 'palmers', 'skin'), 552),
     (('fred', 'palmers', 'skin', 'whitener'), 544),
     (('for', 'over', '40', 'years'), 536),
     (('white', 'bleaching', 'cream', 'as'), 421),
     (('bleaching', 'cream', 'as', 'directed'), 421),
     (('easy', 'and', 'pleasant', 'to'), 382),
     (('use', 'persulan', 'proved', 'more'), 375),
     (('persulan', 'proved', 'more', 'than'), 375),
     (('proved', 'more', 'than', '20'), 375),
     (('more', 'than', '20', 'million'), 375),
     (('than', '20', 'million', 'times'), 375),
     (('20', 'million', 'times', 'everyone'), 375),
     (('million', 'times', 'everyone', 'knows'), 375),
     (('times', 'everyone', 'knows', 'you'), 375),
     (('everyone', 'knows', 'you', 'care'), 375),
     (('knows', 'you', 'care', 'enough'), 375),
     (('you', 'care', 'enough', 'to'), 375),
     (('care', 'enough', 'to', 'use'), 375),
     (('enough', 'to', 'use', 'the'), 375),
     (('to', 'use', 'the', 'finest'), 375),
     (('prescribed', 'by', 'doctors', 'for'), 374),
     (('helps', 'keep', 'skin', 'soft'), 373),
     (('keep', 'skin', 'soft', 'and'), 373),
     (('skin', 'soft', 'and', 'smootha'), 373),
     (('soft', 'and', 'smootha', 'powder'), 373),
     (('and', 'smootha', 'powder', 'basenot'), 373),
     (('smootha', 'powder', 'basenot', 'greasyvanishes'), 373),
     (('powder', 'basenot', 'greasyvanishes', 'immediatelyfor'), 373),
     (('basenot', 'greasyvanishes', 'immediatelyfor', 'skin'), 373),
     (('greasyvanishes', 'immediatelyfor', 'skin', 'blemishes'), 373),
     (('immediatelyfor', 'skin', 'blemishes', 'contains'), 373),
     (('skin', 'blemishes', 'contains', 'the'), 373),
     (('blemishes', 'contains', 'the', 'new'), 373),
     (('contains', 'the', 'new', 'wonderful'), 373),
     (('the', 'new', 'wonderful', 'ingredient'), 373),
     (('new', 'wonderful', 'ingredient', 'persu'), 373),
     (('wonderful', 'ingredient', 'persu', 'persu'), 373),
     (('ingredient', 'persu', 'persu', 'is'), 373),
     (('persu', 'persu', 'is', 'drake'), 373),
     (('persu', 'is', 'drake', 'laboratories'), 373),
     (('is', 'drake', 'laboratories', 'inc'), 373),
     (('drake', 'laboratories', 'inc', 'trademark'), 373),
     (('laboratories', 'inc', 'trademark', 'for'), 373),
     (('inc', 'trademark', 'for', 'stabilized'), 373),
     (('trademark', 'for', 'stabilized', 'hydroquinone'), 373),
     (('for', 'stabilized', 'hydroquinone', 'compounds'), 373),
     (('palmers', 'skin', 'whitener', 'an'), 351),
     (('skin', 'whitener', 'an', 'exclusive'), 351),
     (('whitener', 'an', 'exclusive', 'formula'), 351),
     (('effective', 'ingredients', 'repeatedly', 'prescribed'), 351),
     (('ingredients', 'repeatedly', 'prescribed', 'by'), 351),
     (('repeatedly', 'prescribed', 'by', 'doctors'), 351),
     (('by', 'doctors', 'for', 'skin'), 351),
     (('doctors', 'for', 'skin', 'care'), 351),
     (('as', 'directed', 'and', 'see'), 341),
     (('nadinola', 'deluxe', 'bleaching', 'cream'), 338),
     (('blackheads', 'and', 'refining', 'enlarged'), 337),
     (('and', 'refining', 'enlarged', 'pores'), 337),
     (('refining', 'enlarged', 'pores', 'this'), 337),
     (('enlarged', 'pores', 'this', 'treatment'), 337),
     (('start', 'using', 'black', 'and'), 332),
     (('using', 'black', 'and', 'white'), 332),
     (('safe', 'for', 'normal', 'skin'), 330),
     (('cream', 'as', 'directed', 'and'), 328),
     (('inside', 'your', 'skin', 'modern'), 324),
     (('your', 'skin', 'modern', 'science'), 324),
     (('skin', 'modern', 'science', 'knows'), 324),
     (('directed', 'and', 'see', 'your'), 314),
     (('look', 'its', 'bleaching', 'action'), 312),
     (('bleaching', 'action', 'works', 'effectively'), 311),
     (('men', 'women', 'and', 'young'), 307)]



```python
# A plot of ebony quadgrams
ebony_quadgram_freqs.plot(35, title="Ebony Magazine Quadgrams")
```


    
![png](output_64_0.png)
    





    <AxesSubplot:title={'center':'Ebony Magazine Quadgrams'}, xlabel='Samples', ylabel='Counts'>




```python
# Notice how the concordance for nomral shifts between "use on normal skin" or "for normal skin"
# carries a double meaning "for normal skin." The desire for "normal" skin collides with the
# assertion the product is safe because to admit otherwise would also admit that one's skin
# is somehow not normal. 
ebony_text_object.concordance("normal",120,200)
```

    Displaying 200 of 664 matches:
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too artras deepdown pene
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too artras deepdown pene
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too artras deepdown pene
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too artras deepdown pene
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too artras deepdown pene
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin now lovelier lighter skin safelywuth artra skin ton
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too now lovelier lighter sk
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin have you got that lighter lovelier artra look now a
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too have you got that light
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin soft skin clear skin smooth skin lighter skin lovel
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too soft skin clear skin sm
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    one cream developed by doctors proven effective safe for normal skin with gentle laboratory tested hydroquinone the blea
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    one cream developed by doctors proven effective safe for normal skin that velvetysoft radiant glow that says here is a w
    ns hydroquinone the bleaching ingredient proven safe for normal skin look lighterwith new artra skin tone cream develope
    one cream developed by doctors proven effective safe for normal skin now at last a complexion cream with combined action
    one cream developed by doctors proven effective safe for normal skin long for the radiant glow of lighter brighter skin 
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too look lighterwith new ar
    one cream developed by doctors proven effective safe for normal skin no more messy oldfashioned methods to get the light
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    one cream developed by doctors proven effective safe for normal skin clothes do the most for the girl with a lovely comp
    one cream developed by doctors proven effective safe for normal skin photo caption now one activated skin cream performs
    one cream developed by doctors proven effective safe for normal skin a delightful white cream that makes skin look fresh
    one cream developed by doctors proven effective safe for normal skin if you want to be pretty and popular begin with you
    one cream developed by doctors proven effective safe for normal skin breathless enchanting memorable beauty is your desi
    one cream developed by doctors proven effective safe for normal skin if your skin doesnt look actually lighter after usi
    one cream developed by doctors proven effective safe for normal skin none look lighterwith new artra skin tone cream dev
    one cream developed by doctors proven effective safe for normal skin if your skin doesnt look actually lighter after usi
    one cream developed by doctors proven effective safe for normal skin no man can resist the allure of a lovely complexion
    one cream developed by doctors proven effective safe for normal skin now a complexion cream for lighter lovelier skin th
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    one cream developed by doctors proven effective safe for normal skin if you face has a shiny oil cup look take advantage
    one cream developed by doctors proven effective safe for normal skin a delightful cream that lightens weathered skin mak
    one cream developed by doctors proven effective safe for normal skin amazing new bleach and glow cream lightens brighten
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin now you can be lighter lovelier thanks to the new w
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too now you can be lighter 
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin fashion forecast lighter brighter skin now at last 
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too fashion forecast lighte
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin i am excited about a new good looks cream for the w
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too i am excited about a ne
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin younger looking fresh skin beauty lightens clears s
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too younger looking fresh s
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin look how men flock around the girl with the clear b
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too look how men flock arou
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin the artra promiselighter lovelier skin beauty for y
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too the artra promiselighte
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin mercolized wax cream guarantees lighter looking ski
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too mercolized wax cream gu
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin none now at last a complexion cream with combined a
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too none no more messy oldf
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin mercolized wax cream guarantees lighter looking ski
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too mercolized wax cream gu
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin sure attraction lighter brighter skin now at last a
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too sure attraction lighter
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin now a lovelier lighter skin for youthanks to this w
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too now a lovelier lighter 
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin do big pores blackheads oily skin spoil your looks 
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too do big pores blackheads
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin gives older ladies younger looking skin lightens cl
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too gives older ladies youn
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
    ry tested hydroquinone the bleaching ingredient safe for normal skin artra not only lightens it creams your skin to clea
    ns hydroquinone the bleaching ingredient proven safe for normal skin lighter fairer lovelier skin now at last a complexi
    oquinone the miracleaction bleaching ingredient safe for normal skin artra softens your skin too lighter fairer lovelier
    re artra lightens softens vanishes and artra is safe for normal skin because it contains the bleaching ingredient hydroq
    ient laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too deep down penetratin
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
     you artra with gentle thorough deepdown action safe for normal skin the cream you smooth on in seconds with no timecons
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 
    tras gentle thorough deepdown action completely safe for normal skin so quick you smooth it on in seconds nonoily artra 



```python
# Most common words in context for with keyword 'normal'
ebony_concord_list_obj = ebony_text_object.concordance_list("normal", 50)
ebony_concord_list = []
for con in ebony_concord_list_obj:
    sent = "".join(list(con[4:]))
    ebony_concord_list.append(sent)
ebony_concord_list_freqs = nltk.FreqDist(ebony_concord_list)
ebony_concord_list_freqs.most_common(10)
```




    [('eness and safe use onskin so soothing and eness and safe use on normal skin so soothing and ',
      8),
     ('g ingredient safe forskin artra not only lg ingredient safe for normal skin artra not only l',
      4),
     ('g ingredient safe forskin artra softens yog ingredient safe for normal skin artra softens yo',
      3),
     ('and artra is safe forskin because it contaand artra is safe for normal skin because it conta',
      3),
     ('dient proven safe forskin now lovelier ligdient proven safe for normal skin now lovelier lig',
      1),
     ('dient proven safe forskin have you got thadient proven safe for normal skin have you got tha',
      1),
     ('dient proven safe forskin soft skin clear dient proven safe for normal skin soft skin clear ',
      1),
     ('en effective safe forskin with gentle laboen effective safe for normal skin with gentle labo',
      1),
     ('en effective safe forskin that velvetysoften effective safe for normal skin that velvetysoft',
      1),
     ('dient proven safe forskin look lighterwithdient proven safe for normal skin look lighterwith',
      1)]




```python
ebony_text_object.concordance("modern",120,200)
```

    Displaying 200 of 542 matches:
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin mercolize
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin lighter b
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin life is m
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin dr fred p
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin egyptian 
     its bleachign action works effectively inside your skin modern science knows no faster way of lightening skin now a lov
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     its bleachign action works effectively inside your skin modern science knows no faster way of lightening skin lighter f
     its bleachign action works effectively inside your skin modern science knows no faster way of lightening skin mercolize
     its bleachign action works effectively inside your skin modern science knows no faster way of lightening skin the prett
     its bleachign action works effectively inside your skin modern science knows no faster way of lightening skin get 2 wor
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin get 2 
    zing beauty benefits for just 1 thanks to the miracle of modern science we now know that below the surface of the skin i
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin mercol
    in in just 7 days or money back thanks to the miracle of modern science we now know that below the surface of the skin i
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin be lov
    oved have lighter brighter skin thanks to the miracle of modern science we now know that below the surface of the skin i
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin youd n
    ch lies waiting under your skin thanks to the miracle of modern science we now know that below the surface of the skin i
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin now lovel
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin now lo
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin have you 
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin have y
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin soft skin
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin soft s
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin look ligh
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin look l
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin now you c
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin now yo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     you can be lighter lovelier thanks to the new wonder of modern science artra skin tone cream get that radiant artra loo
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin fashion f
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin fashio
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin i am exci
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin i am e
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin younger l
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin younge
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin look how 
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin look h
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin the artra
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin the ar
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin mercolize
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin mercol
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin none phot
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin none n
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin mercolize
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin mercol
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin sure attr
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin sure a
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin now a lov
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin now a 
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     a lovelier lighter skin for youthanks to this wonder of modern science new artra skin tone cream with miracleaction hyd
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin do big po
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin do big
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin gives old
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin gives 
     for her skin new artra skin tone cream was developed in modern laboratories to do just that to help you care for your s
    one cream developed by doctors is the completely new and modern way to gentle thorough beauty care artra lightens soften
     its bleaching action works effectively inside your skin modern science knows of no faster way to lighten skin lighter f
     caption now one activated skin cream performs all these modern miracles makes face hands look lighter softer youngergua
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    moother look its bleaching action works inside your skin modern science knows of no faster way of lightening skin lighte
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    hanting memorable beauty is your desire a new miracle of modern science now opens the door to a lighter lovelier you dev
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin start usi
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin start usi
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin start usi
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin start usi
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin start usi
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin start usi
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin he never 
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin lighter f
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin thank goo
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin youll be 
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin the artra
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin mercolize
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin use persu
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin posners n
    rable beauty is your desire then listen a new miracle of modern science now opens the door to a lighter lovelier you dev
     its bleaching action works effectively inside your skin modern science knows no faster way of lightening skin thank goo


<span id="terms_by_year"></span>
## Term Counts By Year
This section gives a sense of the shape and distribution of the data. We look at the distribution of terms of interest to the research team and compare their relative uses over time. 


```python
#top twenty terms for Drum
terms_drum = [term[0] for term in drum_most_common_counts[:20]] 
print(terms_drum)
```

    ['skin', 'complexion', 'light', 'karroo', 'lighter', 'ambi', 'beautiful', 'lovely', 'beauty', 'makes', 'extra', 'cream', 'use', 'new', 'look', 'smooth', 'lightening', 'fast', 'night', 'lightens']



```python
#top twenty terms in Ebony
terms_ebony = [term[0] for term in ebony_most_common_counts[:20]] 
print(terms_ebony)
```

    ['skin', 'cream', 'complexion', 'lighter', 'new', 'beauty', 'nadinola', 'dark', 'artra', 'use', 'look', 'glow', 'spots', 'brighter', 'tone', 'blackheads', 'bleach', 'bleaching', 'face', 'see']



```python
terms = set(terms_drum + terms_ebony)
print(f"Of the 40 top terms from Ebony and Drum, there are {len(terms)} unique most frequent terms to plot by year.")
print()
print(terms)
```

    Of the 40 top terms from Ebony and Drum, there are 32 unique most frequent terms to plot by year.
    
    {'ambi', 'nadinola', 'new', 'tone', 'dark', 'extra', 'brighter', 'light', 'beautiful', 'bleaching', 'see', 'lovely', 'use', 'blackheads', 'look', 'skin', 'night', 'smooth', 'spots', 'cream', 'lightens', 'artra', 'bleach', 'beauty', 'face', 'fast', 'karroo', 'makes', 'lightening', 'complexion', 'lighter', 'glow'}



```python
# Drum sentence tuples with date
claims_drum_sents = []
i = 0
for claim in claims_drum:
    temp_claim = "".join([ch for ch in claim if (ch.isalpha() or ch == " ") and ch not in string.punctuation]).lower()
    claims_drum_sents.append((str(claims_drum.keys()[i]),temp_claim))
    i += 1
print(claims_drum_sents[:2])
print(claims_drum_sents[-2:])
```

    [('1965-01-01 00:00:00', 'to make their skin lighter and lovelierlovelier and lightera little more every day american scientist made artralightens and'), ('1965-01-01 00:00:00', 'brightens skin lightens from the first day vanishes into skin instantlystarts working starts lightening and brightening your skin')]
    [('1987-11-01 00:00:00', 'trust heman to bring you a new fresh skin lightening lotion that really works  it gives you the light golden complexion that todays modern'), ('1987-11-01 00:00:00', 'successful women demand ready to help your skin gain the complexion you had  before the harsh south african sun changed it')]



```python
# Ebony sentence tuples with date
claims_ebony_sents = []
i = 0
for claim in claims_ebony:
    temp_claim = "".join([ch for ch in claim if (ch.isalpha() or ch == " ") and ch not in string.punctuation]).lower()
    claims_ebony_sents.append((str(claims_ebony.keys()[i]),temp_claim))
    i += 1
print(claims_ebony_sents[:2])
print(claims_ebony_sents[-2:])
```

    [('1960-01-01 00:00:00', '  wakes up dark dull complexion conceals ugly blotches blemishes while it bleaches guarantees lovelier lighter skin'), ('1960-01-01 00:00:00', 'if your skin doesnt look actually lighter after using mercolized wax cream for just one week your money will be cheerfully refunded youll see amazing results almost at once  as mercolized wax creams speedy bleaching action lightens your complexion fades dark blotches spots and freckles brings excessive skin oiliness under control works under the skin surface to bring about these marvelous results used by beautiful women for over  years')]
    [('1990-12-01 00:00:00', 'created especially for you formula  fades dark marks lines age spots acne marks this remarkable cosmetic lightens knees elbows necksafely quickly and easily use it for stretchmarks freckles scars dont accept imitations'), ('1990-12-01 00:00:00', 'clear smooth new skin with safe effective  day peeling treatment treats acne scars pimples age lines dry blotchy skin razor bumps over  years of success peelerpak home treatment')]



```python
# collect a list of years of publication, in this case for Ebony
# because our data spans from 1960 to 1990.

years = list(sorted(set([date[:4] for date, sent in claims_ebony_sents])))
print(years)
```

    ['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990']



```python
# term counts from ad claims by year (1960-1990) and publication.

drum_term_counts = {}
ebony_term_counts = {}
combined_term_counts = {}
for term in terms:
    temp_term = []
    for year in years:
        e = 0
        for y,c in claims_ebony_sents:
            if str(year) == y[:4]:
                if term in c.split():
                    e += 1
        temp_term.append(e)
    ebony_term_counts[term] = temp_term
    combined_term_counts[term + " (Ebony)"] = temp_term # Added Ebony tag
    temp_term = []
    for year in years:
        e = 0
        for y,c in claims_drum_sents:
            if str(year) == y[:4]:
                if term in c.split():
                    e += 1
        temp_term.append(e)
    drum_term_counts[term] = temp_term
    combined_term_counts[term + " (Drum)"] = temp_term # Added Drum tag
combined_term_counts['years'] = years
print("Ebony Term Counts")
print(ebony_term_counts)
print("-"*100)
print("Drum Term Counts")
print(drum_term_counts)
print("-"*100)
print("Combined Term Counts")
print(combined_term_counts)
```

    Ebony Term Counts
    {'ambi': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 6, 5, 6, 6, 4, 0, 7, 7, 6, 5, 8, 11, 5, 0], 'nadinola': [11, 14, 14, 13, 19, 17, 13, 9, 7, 5, 9, 4, 3, 0, 0, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0], 'new': [26, 54, 35, 42, 43, 25, 19, 8, 6, 10, 28, 14, 0, 0, 0, 0, 0, 8, 4, 6, 6, 5, 2, 0, 1, 4, 7, 12, 18, 25, 13], 'tone': [9, 10, 3, 0, 16, 14, 6, 1, 3, 11, 17, 13, 4, 7, 11, 8, 14, 20, 10, 13, 12, 11, 12, 18, 8, 10, 16, 16, 15, 15, 6], 'dark': [25, 40, 19, 10, 17, 8, 10, 11, 11, 15, 20, 12, 3, 8, 15, 8, 15, 18, 11, 17, 17, 14, 12, 15, 8, 8, 11, 12, 18, 24, 20], 'extra': [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0], 'brighter': [19, 24, 35, 31, 41, 20, 15, 10, 3, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 2, 3, 5, 2, 0, 0, 2, 0, 0, 0, 0, 0], 'light': [13, 24, 7, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 3, 1, 2, 1, 0, 0, 0, 1, 2, 0, 0, 0, 2, 1], 'beautiful': [14, 20, 5, 5, 2, 10, 3, 3, 2, 7, 1, 2, 4, 5, 5, 5, 6, 14, 11, 9, 7, 8, 5, 2, 1, 2, 2, 10, 12, 2, 5], 'bleaching': [36, 47, 27, 18, 17, 18, 15, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'see': [35, 51, 28, 24, 31, 13, 14, 7, 1, 6, 8, 0, 0, 0, 0, 11, 16, 14, 15, 18, 12, 8, 2, 7, 6, 2, 4, 2, 0, 0, 1], 'lovely': [16, 24, 17, 7, 3, 2, 10, 2, 0, 6, 3, 1, 0, 0, 0, 0, 0, 5, 3, 6, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 0], 'use': [20, 35, 22, 20, 17, 21, 28, 15, 9, 17, 20, 7, 2, 3, 8, 4, 4, 10, 5, 12, 12, 17, 9, 5, 5, 11, 8, 1, 8, 16, 14], 'blackheads': [16, 30, 29, 38, 28, 23, 30, 18, 14, 15, 24, 13, 3, 5, 8, 3, 0, 0, 0, 5, 4, 9, 6, 5, 5, 6, 7, 0, 0, 0, 0], 'look': [36, 42, 28, 17, 31, 22, 25, 25, 13, 14, 6, 6, 2, 0, 2, 9, 17, 21, 9, 6, 9, 6, 2, 0, 0, 3, 6, 4, 7, 2, 1], 'skin': [71, 103, 69, 79, 72, 58, 53, 35, 29, 31, 35, 24, 10, 9, 15, 22, 34, 39, 28, 31, 29, 30, 20, 24, 14, 20, 23, 34, 42, 35, 25], 'night': [3, 5, 5, 8, 2, 8, 10, 5, 5, 6, 9, 3, 0, 0, 0, 6, 7, 0, 1, 2, 3, 0, 0, 7, 6, 2, 4, 3, 3, 0, 0], 'smooth': [17, 30, 12, 9, 19, 8, 6, 5, 5, 5, 21, 18, 5, 4, 9, 6, 4, 11, 3, 9, 4, 3, 10, 7, 2, 4, 11, 27, 26, 21, 13], 'spots': [21, 30, 6, 9, 23, 7, 11, 15, 11, 13, 28, 16, 1, 2, 10, 4, 4, 15, 9, 16, 14, 15, 6, 9, 6, 6, 9, 8, 18, 24, 20], 'cream': [50, 73, 42, 44, 49, 32, 34, 17, 16, 23, 28, 23, 5, 6, 5, 12, 22, 22, 7, 15, 14, 9, 10, 16, 8, 12, 14, 21, 26, 16, 8], 'lightens': [24, 28, 12, 9, 9, 4, 6, 3, 4, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 5], 'artra': [12, 8, 13, 9, 8, 4, 3, 2, 1, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 4, 2, 3, 2, 0, 0, 0, 3, 6, 6, 0], 'bleach': [12, 26, 15, 10, 7, 7, 7, 4, 8, 15, 20, 16, 4, 6, 5, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 1, 0, 0], 'beauty': [22, 29, 25, 29, 31, 29, 19, 14, 9, 12, 21, 15, 6, 4, 6, 10, 7, 2, 2, 1, 9, 7, 5, 2, 0, 6, 8, 4, 14, 12, 9], 'face': [7, 23, 12, 9, 24, 22, 16, 14, 10, 12, 6, 4, 2, 2, 6, 6, 19, 30, 24, 23, 20, 15, 2, 8, 6, 5, 8, 1, 8, 11, 10], 'fast': [8, 16, 6, 2, 0, 3, 2, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0], 'karroo': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'makes': [12, 8, 6, 17, 15, 11, 10, 4, 6, 8, 5, 8, 2, 0, 0, 0, 0, 0, 0, 4, 8, 6, 5, 4, 2, 2, 2, 1, 0, 0, 0], 'lightening': [9, 10, 11, 2, 17, 6, 6, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'complexion': [40, 51, 40, 38, 31, 27, 25, 19, 14, 14, 14, 5, 4, 0, 8, 13, 20, 11, 13, 13, 15, 17, 9, 12, 12, 13, 16, 4, 5, 7, 2], 'lighter': [45, 52, 56, 47, 50, 33, 26, 15, 12, 8, 2, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'glow': [9, 18, 17, 26, 23, 14, 7, 4, 3, 4, 6, 7, 4, 6, 5, 4, 7, 8, 6, 9, 8, 3, 3, 4, 0, 2, 4, 2, 8, 8, 5]}
    ----------------------------------------------------------------------------------------------------
    Drum Term Counts
    {'ambi': [0, 0, 0, 0, 0, 4, 3, 5, 0, 0, 12, 2, 6, 1, 3, 7, 18, 11, 5, 4, 1, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0], 'nadinola': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'new': [0, 0, 0, 0, 0, 0, 0, 3, 6, 3, 6, 10, 1, 16, 0, 0, 1, 0, 0, 0, 0, 3, 0, 2, 0, 1, 3, 3, 0, 0, 0], 'tone': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0], 'dark': [0, 0, 0, 0, 0, 0, 6, 12, 3, 8, 14, 9, 0, 9, 8, 8, 8, 6, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0], 'extra': [0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 17, 6, 7, 1, 0, 0, 2, 2, 0, 0, 3, 4, 6, 10, 3, 2, 2, 6, 0, 0, 0], 'brighter': [0, 0, 0, 0, 0, 0, 3, 5, 4, 4, 3, 10, 0, 0, 0, 2, 0, 0, 0, 0, 5, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0], 'light': [0, 0, 0, 0, 0, 9, 9, 9, 17, 12, 17, 11, 28, 15, 11, 14, 3, 5, 1, 0, 5, 0, 3, 5, 3, 1, 5, 9, 0, 0, 0], 'beautiful': [0, 0, 0, 0, 0, 4, 6, 6, 0, 3, 3, 2, 19, 23, 19, 13, 13, 10, 3, 2, 8, 3, 6, 17, 0, 0, 0, 0, 0, 0, 0], 'bleaching': [0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'see': [0, 0, 0, 0, 0, 1, 2, 1, 4, 4, 8, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lovely': [0, 0, 0, 0, 0, 5, 1, 8, 15, 17, 11, 9, 21, 10, 5, 11, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'use': [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 4, 10, 10, 0, 0, 0, 2, 2, 4, 1, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0], 'blackheads': [0, 0, 0, 0, 0, 0, 5, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'look': [0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 6, 0, 6, 15, 0, 2, 1, 0, 1, 1, 3, 4, 5, 14, 4, 0, 0, 0, 0, 0, 0], 'skin': [0, 0, 0, 0, 0, 39, 33, 53, 65, 81, 75, 59, 45, 59, 39, 55, 36, 36, 6, 11, 9, 14, 18, 35, 11, 5, 8, 13, 0, 0, 0], 'night': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 3, 7, 12, 9, 6, 0, 3, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'smooth': [0, 0, 0, 0, 0, 6, 11, 13, 14, 27, 8, 9, 21, 16, 15, 23, 13, 6, 1, 2, 1, 1, 8, 15, 1, 0, 2, 2, 0, 0, 0], 'spots': [0, 0, 0, 0, 0, 1, 5, 11, 11, 15, 16, 5, 10, 19, 13, 13, 10, 5, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'cream': [0, 0, 0, 0, 0, 2, 1, 8, 7, 6, 21, 12, 1, 5, 5, 8, 0, 3, 0, 0, 1, 1, 0, 2, 4, 3, 0, 2, 0, 0, 0], 'lightens': [0, 0, 0, 0, 0, 5, 1, 0, 8, 4, 7, 22, 5, 38, 17, 29, 16, 15, 0, 0, 0, 0, 3, 3, 3, 1, 2, 2, 0, 0, 0], 'artra': [0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bleach': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'beauty': [0, 0, 0, 0, 0, 0, 0, 3, 10, 3, 3, 4, 10, 1, 0, 0, 1, 2, 0, 3, 8, 11, 4, 7, 3, 0, 0, 0, 0, 0, 0], 'face': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 10, 9, 8, 10, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'fast': [0, 0, 0, 0, 0, 0, 5, 9, 12, 14, 27, 13, 15, 0, 1, 7, 10, 11, 2, 4, 5, 5, 6, 12, 3, 2, 2, 6, 0, 0, 0], 'karroo': [0, 0, 0, 0, 0, 2, 7, 6, 0, 3, 6, 8, 12, 13, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'makes': [0, 0, 0, 0, 0, 6, 2, 2, 7, 9, 10, 11, 16, 8, 1, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'lightening': [0, 0, 0, 0, 0, 6, 9, 8, 0, 0, 7, 2, 0, 2, 0, 0, 14, 12, 0, 1, 3, 4, 3, 7, 4, 2, 5, 5, 0, 0, 0], 'complexion': [0, 0, 0, 0, 0, 18, 18, 31, 29, 47, 33, 17, 29, 14, 14, 12, 18, 20, 2, 0, 7, 7, 4, 14, 0, 1, 6, 6, 0, 0, 0], 'lighter': [0, 0, 0, 0, 0, 22, 17, 25, 27, 48, 26, 33, 14, 11, 18, 17, 16, 12, 2, 2, 1, 4, 1, 2, 6, 0, 0, 0, 0, 0, 0], 'glow': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    ----------------------------------------------------------------------------------------------------
    Combined Term Counts
    {'ambi (Ebony)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 6, 5, 6, 6, 4, 0, 7, 7, 6, 5, 8, 11, 5, 0], 'ambi (Drum)': [0, 0, 0, 0, 0, 4, 3, 5, 0, 0, 12, 2, 6, 1, 3, 7, 18, 11, 5, 4, 1, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0], 'nadinola (Ebony)': [11, 14, 14, 13, 19, 17, 13, 9, 7, 5, 9, 4, 3, 0, 0, 2, 1, 4, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0], 'nadinola (Drum)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'new (Ebony)': [26, 54, 35, 42, 43, 25, 19, 8, 6, 10, 28, 14, 0, 0, 0, 0, 0, 8, 4, 6, 6, 5, 2, 0, 1, 4, 7, 12, 18, 25, 13], 'new (Drum)': [0, 0, 0, 0, 0, 0, 0, 3, 6, 3, 6, 10, 1, 16, 0, 0, 1, 0, 0, 0, 0, 3, 0, 2, 0, 1, 3, 3, 0, 0, 0], 'tone (Ebony)': [9, 10, 3, 0, 16, 14, 6, 1, 3, 11, 17, 13, 4, 7, 11, 8, 14, 20, 10, 13, 12, 11, 12, 18, 8, 10, 16, 16, 15, 15, 6], 'tone (Drum)': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0], 'dark (Ebony)': [25, 40, 19, 10, 17, 8, 10, 11, 11, 15, 20, 12, 3, 8, 15, 8, 15, 18, 11, 17, 17, 14, 12, 15, 8, 8, 11, 12, 18, 24, 20], 'dark (Drum)': [0, 0, 0, 0, 0, 0, 6, 12, 3, 8, 14, 9, 0, 9, 8, 8, 8, 6, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 0], 'extra (Ebony)': [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 1, 0, 0, 0], 'extra (Drum)': [0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 17, 6, 7, 1, 0, 0, 2, 2, 0, 0, 3, 4, 6, 10, 3, 2, 2, 6, 0, 0, 0], 'brighter (Ebony)': [19, 24, 35, 31, 41, 20, 15, 10, 3, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 2, 3, 5, 2, 0, 0, 2, 0, 0, 0, 0, 0], 'brighter (Drum)': [0, 0, 0, 0, 0, 0, 3, 5, 4, 4, 3, 10, 0, 0, 0, 2, 0, 0, 0, 0, 5, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0], 'light (Ebony)': [13, 24, 7, 4, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 3, 1, 2, 1, 0, 0, 0, 1, 2, 0, 0, 0, 2, 1], 'light (Drum)': [0, 0, 0, 0, 0, 9, 9, 9, 17, 12, 17, 11, 28, 15, 11, 14, 3, 5, 1, 0, 5, 0, 3, 5, 3, 1, 5, 9, 0, 0, 0], 'beautiful (Ebony)': [14, 20, 5, 5, 2, 10, 3, 3, 2, 7, 1, 2, 4, 5, 5, 5, 6, 14, 11, 9, 7, 8, 5, 2, 1, 2, 2, 10, 12, 2, 5], 'beautiful (Drum)': [0, 0, 0, 0, 0, 4, 6, 6, 0, 3, 3, 2, 19, 23, 19, 13, 13, 10, 3, 2, 8, 3, 6, 17, 0, 0, 0, 0, 0, 0, 0], 'bleaching (Ebony)': [36, 47, 27, 18, 17, 18, 15, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bleaching (Drum)': [0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'see (Ebony)': [35, 51, 28, 24, 31, 13, 14, 7, 1, 6, 8, 0, 0, 0, 0, 11, 16, 14, 15, 18, 12, 8, 2, 7, 6, 2, 4, 2, 0, 0, 1], 'see (Drum)': [0, 0, 0, 0, 0, 1, 2, 1, 4, 4, 8, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lovely (Ebony)': [16, 24, 17, 7, 3, 2, 10, 2, 0, 6, 3, 1, 0, 0, 0, 0, 0, 5, 3, 6, 0, 0, 0, 0, 0, 0, 0, 3, 4, 2, 0], 'lovely (Drum)': [0, 0, 0, 0, 0, 5, 1, 8, 15, 17, 11, 9, 21, 10, 5, 11, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'use (Ebony)': [20, 35, 22, 20, 17, 21, 28, 15, 9, 17, 20, 7, 2, 3, 8, 4, 4, 10, 5, 12, 12, 17, 9, 5, 5, 11, 8, 1, 8, 16, 14], 'use (Drum)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 4, 10, 10, 0, 0, 0, 2, 2, 4, 1, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0], 'blackheads (Ebony)': [16, 30, 29, 38, 28, 23, 30, 18, 14, 15, 24, 13, 3, 5, 8, 3, 0, 0, 0, 5, 4, 9, 6, 5, 5, 6, 7, 0, 0, 0, 0], 'blackheads (Drum)': [0, 0, 0, 0, 0, 0, 5, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'look (Ebony)': [36, 42, 28, 17, 31, 22, 25, 25, 13, 14, 6, 6, 2, 0, 2, 9, 17, 21, 9, 6, 9, 6, 2, 0, 0, 3, 6, 4, 7, 2, 1], 'look (Drum)': [0, 0, 0, 0, 0, 0, 4, 7, 0, 0, 6, 0, 6, 15, 0, 2, 1, 0, 1, 1, 3, 4, 5, 14, 4, 0, 0, 0, 0, 0, 0], 'skin (Ebony)': [71, 103, 69, 79, 72, 58, 53, 35, 29, 31, 35, 24, 10, 9, 15, 22, 34, 39, 28, 31, 29, 30, 20, 24, 14, 20, 23, 34, 42, 35, 25], 'skin (Drum)': [0, 0, 0, 0, 0, 39, 33, 53, 65, 81, 75, 59, 45, 59, 39, 55, 36, 36, 6, 11, 9, 14, 18, 35, 11, 5, 8, 13, 0, 0, 0], 'night (Ebony)': [3, 5, 5, 8, 2, 8, 10, 5, 5, 6, 9, 3, 0, 0, 0, 6, 7, 0, 1, 2, 3, 0, 0, 7, 6, 2, 4, 3, 3, 0, 0], 'night (Drum)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 3, 7, 12, 9, 6, 0, 3, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'smooth (Ebony)': [17, 30, 12, 9, 19, 8, 6, 5, 5, 5, 21, 18, 5, 4, 9, 6, 4, 11, 3, 9, 4, 3, 10, 7, 2, 4, 11, 27, 26, 21, 13], 'smooth (Drum)': [0, 0, 0, 0, 0, 6, 11, 13, 14, 27, 8, 9, 21, 16, 15, 23, 13, 6, 1, 2, 1, 1, 8, 15, 1, 0, 2, 2, 0, 0, 0], 'spots (Ebony)': [21, 30, 6, 9, 23, 7, 11, 15, 11, 13, 28, 16, 1, 2, 10, 4, 4, 15, 9, 16, 14, 15, 6, 9, 6, 6, 9, 8, 18, 24, 20], 'spots (Drum)': [0, 0, 0, 0, 0, 1, 5, 11, 11, 15, 16, 5, 10, 19, 13, 13, 10, 5, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'cream (Ebony)': [50, 73, 42, 44, 49, 32, 34, 17, 16, 23, 28, 23, 5, 6, 5, 12, 22, 22, 7, 15, 14, 9, 10, 16, 8, 12, 14, 21, 26, 16, 8], 'cream (Drum)': [0, 0, 0, 0, 0, 2, 1, 8, 7, 6, 21, 12, 1, 5, 5, 8, 0, 3, 0, 0, 1, 1, 0, 2, 4, 3, 0, 2, 0, 0, 0], 'lightens (Ebony)': [24, 28, 12, 9, 9, 4, 6, 3, 4, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 5], 'lightens (Drum)': [0, 0, 0, 0, 0, 5, 1, 0, 8, 4, 7, 22, 5, 38, 17, 29, 16, 15, 0, 0, 0, 0, 3, 3, 3, 1, 2, 2, 0, 0, 0], 'artra (Ebony)': [12, 8, 13, 9, 8, 4, 3, 2, 1, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 2, 4, 2, 3, 2, 0, 0, 0, 3, 6, 6, 0], 'artra (Drum)': [0, 0, 0, 0, 0, 0, 0, 3, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bleach (Ebony)': [12, 26, 15, 10, 7, 7, 7, 4, 8, 15, 20, 16, 4, 6, 5, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 1, 0, 0], 'bleach (Drum)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'beauty (Ebony)': [22, 29, 25, 29, 31, 29, 19, 14, 9, 12, 21, 15, 6, 4, 6, 10, 7, 2, 2, 1, 9, 7, 5, 2, 0, 6, 8, 4, 14, 12, 9], 'beauty (Drum)': [0, 0, 0, 0, 0, 0, 0, 3, 10, 3, 3, 4, 10, 1, 0, 0, 1, 2, 0, 3, 8, 11, 4, 7, 3, 0, 0, 0, 0, 0, 0], 'face (Ebony)': [7, 23, 12, 9, 24, 22, 16, 14, 10, 12, 6, 4, 2, 2, 6, 6, 19, 30, 24, 23, 20, 15, 2, 8, 6, 5, 8, 1, 8, 11, 10], 'face (Drum)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 10, 9, 8, 10, 9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'fast (Ebony)': [8, 16, 6, 2, 0, 3, 2, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0], 'fast (Drum)': [0, 0, 0, 0, 0, 0, 5, 9, 12, 14, 27, 13, 15, 0, 1, 7, 10, 11, 2, 4, 5, 5, 6, 12, 3, 2, 2, 6, 0, 0, 0], 'karroo (Ebony)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'karroo (Drum)': [0, 0, 0, 0, 0, 2, 7, 6, 0, 3, 6, 8, 12, 13, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'makes (Ebony)': [12, 8, 6, 17, 15, 11, 10, 4, 6, 8, 5, 8, 2, 0, 0, 0, 0, 0, 0, 4, 8, 6, 5, 4, 2, 2, 2, 1, 0, 0, 0], 'makes (Drum)': [0, 0, 0, 0, 0, 6, 2, 2, 7, 9, 10, 11, 16, 8, 1, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'lightening (Ebony)': [9, 10, 11, 2, 17, 6, 6, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lightening (Drum)': [0, 0, 0, 0, 0, 6, 9, 8, 0, 0, 7, 2, 0, 2, 0, 0, 14, 12, 0, 1, 3, 4, 3, 7, 4, 2, 5, 5, 0, 0, 0], 'complexion (Ebony)': [40, 51, 40, 38, 31, 27, 25, 19, 14, 14, 14, 5, 4, 0, 8, 13, 20, 11, 13, 13, 15, 17, 9, 12, 12, 13, 16, 4, 5, 7, 2], 'complexion (Drum)': [0, 0, 0, 0, 0, 18, 18, 31, 29, 47, 33, 17, 29, 14, 14, 12, 18, 20, 2, 0, 7, 7, 4, 14, 0, 1, 6, 6, 0, 0, 0], 'lighter (Ebony)': [45, 52, 56, 47, 50, 33, 26, 15, 12, 8, 2, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'lighter (Drum)': [0, 0, 0, 0, 0, 22, 17, 25, 27, 48, 26, 33, 14, 11, 18, 17, 16, 12, 2, 2, 1, 4, 1, 2, 6, 0, 0, 0, 0, 0, 0], 'glow (Ebony)': [9, 18, 17, 26, 23, 14, 7, 4, 3, 4, 6, 7, 4, 6, 5, 4, 7, 8, 6, 9, 8, 3, 3, 4, 0, 2, 4, 2, 8, 8, 5], 'glow (Drum)': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'years': ['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990']}



```python
# Ensure the shape of the data is consistent

print(len(drum_term_counts['beautiful']))
print(len(ebony_term_counts['beautiful']))
print(len(combined_term_counts['years']))
```

    31
    31
    31



```python
print("Drum Term Counts:")
drum_top_term_counts = []
for term in drum_term_counts:
    drum_term_sum = sum(map(int, drum_term_counts[term]))
    drum_top_term_counts.append((term, drum_term_sum))
for term in reversed(sorted(drum_top_term_counts, key = lambda x: x[1])):
    print("\t", term)
```

    Drum Term Counts:
    	 ('skin', 805)
    	 ('complexion', 347)
    	 ('lighter', 304)
    	 ('smooth', 215)
    	 ('light', 192)
    	 ('lightens', 181)
    	 ('fast', 171)
    	 ('beautiful', 160)
    	 ('spots', 136)
    	 ('lovely', 118)
    	 ('dark', 98)
    	 ('lightening', 94)
    	 ('cream', 92)
    	 ('ambi', 89)
    	 ('makes', 78)
    	 ('extra', 78)
    	 ('karroo', 73)
    	 ('beauty', 73)
    	 ('look', 73)
    	 ('face', 58)
    	 ('new', 58)
    	 ('use', 54)
    	 ('night', 53)
    	 ('brighter', 42)
    	 ('see', 25)
    	 ('blackheads', 12)
    	 ('bleaching', 12)
    	 ('tone', 11)
    	 ('artra', 6)
    	 ('bleach', 4)
    	 ('glow', 3)
    	 ('nadinola', 0)



```python
print("Ebony Term Counts:")
ebony_top_term_counts = []
for term in ebony_term_counts:
    ebony_term_sum = sum(map(int, ebony_term_counts[term]))
    ebony_top_term_counts.append((term, ebony_term_sum))
for term in reversed(sorted(ebony_top_term_counts, key = lambda x: x[1])):
    print("\t", term)
```

    Ebony Term Counts:
    	 ('skin', 1143)
    	 ('cream', 679)
    	 ('complexion', 512)
    	 ('dark', 452)
    	 ('new', 421)
    	 ('spots', 386)
    	 ('use', 385)
    	 ('look', 371)
    	 ('beauty', 369)
    	 ('face', 365)
    	 ('lighter', 351)
    	 ('blackheads', 344)
    	 ('see', 336)
    	 ('smooth', 334)
    	 ('tone', 329)
    	 ('glow', 234)
    	 ('brighter', 222)
    	 ('beautiful', 189)
    	 ('bleaching', 182)
    	 ('bleach', 179)
    	 ('nadinola', 149)
    	 ('makes', 146)
    	 ('lightens', 119)
    	 ('lovely', 114)
    	 ('night', 113)
    	 ('ambi', 94)
    	 ('artra', 93)
    	 ('light', 76)
    	 ('lightening', 67)
    	 ('fast', 52)
    	 ('extra', 15)
    	 ('karroo', 0)



```python
# graph made with Seaborn style in vanilla matplotlib
from matplotlib import cm # color maps import
from random import sample

width = 0.9

custom_cmap = plt.get_cmap('tab20')

fig, ax = plt.subplots(figsize=(20, 15))

color_vals = sample([num for num in range(0,20)], k=20)

counter = 0 

for term in terms_ebony:
    ax.bar(years, 
           ebony_term_counts[term], 
           width, 
           label=term, 
           align="center",
           color=custom_cmap.colors[color_vals[counter]])
    counter += 1
    
ax.set_ylabel("Counts\n")
ax.set_xlabel("\nPublication Year")
ax.set_title("Most Common Terms by Year in Ebony Magazine\n")
ax.set_xticklabels(years,rotation=90)
ax.legend(bbox_to_anchor=(0, -.1), 
          loc='upper left',
          ncol=5, 
          borderaxespad=2)
plt.show()
fig.savefig("most_common_terms_by_year_ebony.png")
```

    <ipython-input-74-53c56dc9f681>:27: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(years,rotation=90)



    
![png](output_79_1.png)
    



```python
# graph made with Seaborn style in vanilla matplotlib

from matplotlib import cm # color maps import
from random import sample

width = 0.9

custom_cmap = plt.get_cmap('tab20')

fig, ax = plt.subplots(figsize=(20, 15))

# color_vals = sample([num for num in range(0,20)], k=20) #omit to keep colors consistent between graphs

counter = 0 

for term in terms_drum:
    ax.bar(years, 
           drum_term_counts[term], 
           width, 
           label=term, 
           align="center",
           color=custom_cmap.colors[color_vals[counter]])
    counter += 1
    
ax.set_ylabel("Counts\n")
ax.set_xlabel("\nPublication Year")
ax.set_title("Most Common Terms by Year in Drum Magazine\n")
ax.set_xticklabels(years,rotation=90)
ax.legend(bbox_to_anchor=(0, -.1), 
          loc='upper left',
          ncol=5, 
          borderaxespad=2)
plt.show()
fig.savefig("most_common_terms_by_year_drum.png")
```

    <ipython-input-75-9bc6c08f05f5>:28: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(years,rotation=90)



    
![png](output_80_1.png)
    



```python
# original stacked bar graphs using PyGal library, disregard for stylistic consistency

line_chart = pygal.StackedBar(x_label_rotation=40, legend_at_bottom=True)
line_chart.title = 'Term Distribution of Skin Bleaching Ads in Drum from 1965 to 1990'
line_chart.x_labels = map(str, range(1965, 1989))
for term in terms:
    line_chart.add(term, drum_term_counts[term])
display(SVG(line_chart.render(disable_xml_declaration=True))) 
line_chart.render_to_file("drum_term_counts_by_year.svg")
```


    
![svg](output_81_0.svg)
    



```python
# original stacked bar graphs using PyGal library, disregard for stylistic consistency

line_chart = pygal.StackedBar(x_label_rotation=40, legend_at_bottom=True)
line_chart.title = 'Term Distribution of Skin Bleaching Ads in Ebony from 1960 to 1990'
line_chart.x_labels = map(str, range(1960, 1991))
for term in terms:
    line_chart.add(term, ebony_term_counts[term])
display(SVG(line_chart.render(disable_xml_declaration=True))) 
line_chart.render_to_file("ebony_term_counts_by_year.svg")
```


    
![svg](output_82_0.svg)
    



```python
drum_counts = []
ebony_counts = []
for year in years_ebony:
    e = 0
    for y,c in claims_ebony_sents:
        if str(year) == y[:4]:
            e += 1
    ebony_counts.append(e)
    d = 0
    for y,c in claims_drum_sents:
        if str(year) == y[:4]:
            d += 1
    drum_counts.append(d)
print(ebony_counts)
print(drum_counts)
```

    [72, 103, 74, 82, 72, 58, 54, 36, 31, 34, 35, 24, 10, 9, 15, 22, 35, 39, 28, 31, 29, 32, 20, 24, 14, 20, 23, 34, 42, 41, 31]
    [0, 0, 0, 0, 0, 46, 62, 93, 108, 132, 134, 96, 80, 99, 60, 82, 65, 68, 6, 12, 22, 28, 25, 57, 15, 7, 10, 17, 0, 0, 0]



```python
# original stacked bar graphs using PyGal library, disregard for stylistic consistency

line_chart = pygal.StackedBar(x_label_rotation=40, legend_at_bottom=True)
line_chart.title = 'Number of Skin Bleaching Ads in Ebony and Drum from 1960 to 1990'
line_chart.x_labels = map(str, range(1960, 1991))
line_chart.add("Ebony Total Ad Counts", ebony_counts)
line_chart.add("Drum Total Ad Counts", drum_counts)
display(SVG(line_chart.render(disable_xml_declaration=True))) 
line_chart.render_to_file("drum_and_ebony_ad_counts_by_year.svg")
```


    
![svg](output_84_0.svg)
    



```python
# original stacked bar graphs using PyGal library, disregard for stylistic consistency

line_chart = pygal.StackedBar(x_label_rotation=40)
line_chart.title = 'Term Occurrence in Ebony Skin Bleaching Ads from 1960 to 1990'
line_chart.x_labels = map(str, range(1960, 1991))
ebony_year_list = []
for year in years_ebony:
    i = 0
    for y,c in claims_ebony_sents:
        if str(year) == y[:4]:
            for term in terms:
                if term in c.split():
                    i += 1
    ebony_year_list.append(i)
line_chart.add("Term Count", ebony_year_list)
display(SVG(line_chart.render(disable_xml_declaration=True))) 
line_chart.render_to_file("ebony_term_occurrence_by_year.svg")
```


    
![svg](output_85_0.svg)
    



```python
claims_drum = claims_drum.dropna()
#claims_drum.replace(',','',regex = True) 
```


```python
# original stacked bar graphs using PyGal library, disregard for stylistic consistency

line_chart = pygal.StackedBar(x_label_rotation=40)
line_chart.title = 'Term Occurrence in Drum Skin Bleaching Ads from 1965 to 1987'
line_chart.x_labels = map(str, range(1965, 1988))
drum_year_list = []
for year in years_drum:
    i = 0
    for y,c in claims_drum_sents:
        if str(year) == y[:4]:
            for term in terms:
                if term in c.split():
                    i += 1
    drum_year_list.append(i)
line_chart.add("Term Count", drum_year_list)
display(SVG(line_chart.render(disable_xml_declaration=True))) 
line_chart.render_to_file("drum_term_occurrence_by_year.svg")
```


    
![svg](output_87_0.svg)
    



```python
fill = [None for n in range(0,5)]
```


```python
# original stacked bar graphs using PyGal library, disregard for stylistic consistency

line_chart = pygal.StackedBar(x_label_rotation=40, legend_at_bottom=True)
line_chart.title = 'Term Occurrence in Skin Bleaching Ads from 1960 to 1990'
line_chart.x_labels = map(str, range(1960, 1991))
line_chart.add("Ebony Term Counts", ebony_year_list)
line_chart.add("Drum Term Counts", fill+drum_year_list)
display(SVG(line_chart.render(disable_xml_declaration=True))) 
line_chart.render_to_file("drum_and_ebony_term_occurrence_by_year.svg")
```


    
![svg](output_89_0.svg)
    



```python
def add(list_counts):
    total = 0
    for i in list_counts:
        total += i
    return total
```


```python
successfig = plt.figure(figsize=(10, 5), dpi=300)
objects = tuple([term.capitalize() for term in terms])
corpus_counts = dict(ebony_term_counts, **drum_term_counts)
termS1 = [add(count) for word,count in corpus_counts.items() if word in terms]
y_pos = np.arange(len(termS1))

plt.barh(y_pos, termS1, align='center', alpha=.5)
plt.yticks(y_pos, objects, fontsize = 10)
plt.xlabel('Count')
plt.title('Corpus Level Term Frequency')
 
plt.show()
successfig.savefig("corpus_freqs.png")
```


    
![png](output_91_0.png)
    



```python
successfig = plt.figure(figsize=(10, 5), dpi=300)
objects = tuple([term.capitalize() for term in terms])
termS1 = [add(count) for word,count in ebony_term_counts.items() if word in terms]
y_pos = np.arange(len(termS1))

plt.barh(y_pos, termS1, align='center', alpha=.5)
plt.yticks(y_pos, objects, fontsize = 10)
plt.xlabel('Count')
plt.title('Ebony Term Frequency (1960-1990)')
 
plt.show()
successfig.savefig("ebony_freqs.png")
```


    
![png](output_92_0.png)
    



```python
successfig = plt.figure(figsize=(10, 5), dpi=300)
objects = tuple([term.capitalize() for term in terms])
termS1 = [add(count) for word,count in drum_term_counts.items() if word in terms]
y_pos = np.arange(len(termS1))

plt.barh(y_pos, termS1, align='center', alpha=.5)
plt.yticks(y_pos, objects, fontsize = 10)
plt.xlabel('Count')
plt.title('Drum Term Frequency (1965-1988)')
 
plt.show()
successfig.savefig("drum_freqs.png")
```


    
![png](output_93_0.png)
    



```python
# data to plot
n_groups = len(objects)
fig = plt.figure(figsize=(800, 550), dpi=300)
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.28
opacity = 0.5

 
rects1 = plt.bar(index, [add(count) for word,count in drum_term_counts.items() if word in terms], bar_width,
                 alpha=opacity,
                 color='b',
                 label='Drum')
 
rects2 = plt.bar(index + bar_width, [add(count) for word,count in ebony_term_counts.items() if word in terms], bar_width,
                 alpha=opacity,
                 color='g',
                 label='Ebony')
 
plt.xlabel('Terms')
plt.ylabel('Count')
plt.title('Compared Term Use')
plt.xticks(index + bar_width, objects)
plt.xticks(rotation=90, size=20)
plt.legend(loc=2, prop={'size': 20})
 
plt.tight_layout()
plt.show()
fig.savefig("compared_term_use.png")
```


    <Figure size 240000x165000 with 0 Axes>



    
![png](output_94_1.png)
    


<span id="pos"></span>
# Parts of Speech


```python
print("Drum Sample")
print(drum_claims_phrase_list[:5])
print("Ebony Sample")
print(ebony_claims_phrase_list[:5])
```

    Drum Sample
    [["Lighter, lovelier skin today…the American way!'", "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and"], ["Lighter, lovelier skin today…the American way!'", "brightens skin.' '…lightens from the first day.' '…vanishes into skin instantly…starts working, starts lightening and brightening your skin"], ["Lighter, lovelier skin today…the American way!'", "immediately.' '…keeps skin beautiful and clean, makes it smooth and lovely"], ["Lighter, lovelier skin today…the American way!'", '…mild and gentle…keeps skin free from blemishes and pimples'], ["Lighter, lovelier skin today…the American way!'", "…clears and lightens the skin, smooth's away blemishes and spots, softens the skin'"]]
    Ebony Sample
    [['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"', '" - wakes up dark, dull complexion! Conceals ugly blotches, blemishes while it bleaches. Guarantees lovelier, lighter skin."'], ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"', '"If your skin doesn\'t look actually lighter after using Mercolized Wax Cream for just one week, your money will be cheerfully refunded."; "You\'ll see amazing results almost at once - as Mercolized Wax Cream\'s speedy bleaching action lightens your complexion, fades dark blotches, spots, and freckles, brings excessive skin oiliness under control."; "...works under the skin surface to bring about these marvelous results."; "Used by beautiful women for over 40 years."'], ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"', '"And you, too, can have a glamorous complexion!"; "…see your skin get a lighter, brighter, softer look."; "Its bleaching action works effectively inside your skin. Modern science knows no faster way of lightening skin."'], ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"', '"Don\'t let dull, dark skin rob you of romance. Don\'t let oiliness, big pores, blackheads cheat you of charm."; "This remarkable medicated ingredient works deep down within the skin to brighten and lighten it…"; "Soon your skin feels smoother and softer, fresh and fascinating, glowing and glamorous."'], ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"', '"Yes in just 7 days be delighted how fast and easy this doctor\'s fomrula lightens, brightens, and  helps clear skin or money back"; "It lightens, brightens and clears skin fast and at the same time fades blemishes, freckles and off-color spots."']]



```python
drum_tags = []
for ad in drum_claims_phrase_list:
    for sent in ad:
        tokens = nltk.word_tokenize(str(sent))
        tagged = nltk.pos_tag(tokens)
        drum_tags.append(tagged)
```


```python
drum_tags_list = list(itertools.chain.from_iterable(drum_tags))
drum_tags_list[:25]
```




    [('Lighter', 'NNP'),
     (',', ','),
     ('lovelier', 'JJR'),
     ('skin', 'FW'),
     ('today…the', 'JJ'),
     ('American', 'JJ'),
     ('way', 'NN'),
     ('!', '.'),
     ("'", "''"),
     ('…to', 'NNS'),
     ('make', 'VBP'),
     ('their', 'PRP$'),
     ('skin', 'NN'),
     ('lighter', 'NN'),
     ('and', 'CC'),
     ('lovelier…lovelier', 'JJR'),
     ('and', 'CC'),
     ('lighter…a', 'VB'),
     ('little', 'RB'),
     ('more', 'RBR'),
     ('every', 'DT'),
     ('day', 'NN'),
     ("'", 'POS'),
     ("'…American", 'JJ'),
     ('scientist', 'NN')]




```python
# In the most common adjects in Drum Magazine, we can confirm many of the terms used previously.
# The assocation of success and modernity, along with "healthy" confirm the attempts to 
# legitimize these products. 
adjectives = [token for token,pos in drum_tags_list if "JJ" in pos]
adjFreqs = nltk.FreqDist(adjectives)
adjFreqs.plot(20, title="Drum Magazine Skin Bleaching Ad POS ('JJ')")
```


    
![png](output_99_0.png)
    





    <AxesSubplot:title={'center':"Drum Magazine Skin Bleaching Ad POS ('JJ')"}, xlabel='Samples', ylabel='Counts'>




```python
# When considering nouns only, it is remarkable that the majority of terms reference 
# more neutral beauty based phrases. The references to Hollywood confirms the connection to a US
# sense of success and beauty.
nouns = [token for token,pos in drum_tags_list if "NN" in pos]
adjFreqs = nltk.FreqDist(nouns)
adjFreqs.plot(20, title="Drum Magazine Skin Bleaching Ad POS ('NN')")
```


    
![png](output_100_0.png)
    





    <AxesSubplot:title={'center':"Drum Magazine Skin Bleaching Ad POS ('NN')"}, xlabel='Samples', ylabel='Counts'>




```python
drum_text_object.concordance("Hollywood",120,200)
```

    Displaying 200 of 1048 matches:
    s clear and light and lovely the strong one for mennew hollywood seven extra extra for men skin as clear and lovely as 
    s without blemishes or spots the strong one for mennew hollywood seven extra extra for men cream which gives the comple
     and new freckle free beauty the strong one for mennew hollywood seven extra extra for men lighter softer smoother comp
    r softer smoother complexion the strong one for mennew hollywood seven extra extra for men if your skin is dark rough o
    imple freckle and complexion the strong one for mennew hollywood seven extra extra for men lotion quick results the str
    for men lotion quick results the strong one for mennew hollywood seven extra extra for men lighter smoother skins does 
    es less of the brown pigment the strong one for mennew hollywood seven extra extra for men makes your skin naturally li
     your skin naturally lighter the strong one for mennew hollywood seven extra extra for men lighter clearer complexion s
    ticed successful people have the strong one for mennew hollywood seven extra extra for men clears the skin of pimples a
    nd lighter within a few days the strong one for mennew hollywood seven extra extra for men double strength skin lighten
    ightener for quicker results the strong one for mennew hollywood seven extra extra for men works fast on all skins the 
     men works fast on all skins the strong one for mennew hollywood seven extra extra for men for a cleaner healthier skin
    for a cleaner healthier skin the strong one for mennew hollywood seven extra extra for men to make your skin softersmoo
    tersmoother than ever before the strong one for mennew hollywood seven extra extra for men men contains extra hydroquin
    s freshens works fast brings the strong one for mennew hollywood seven extra extra for men the good looks the strong on
    extra for men the good looks the strong one for mennew hollywood seven extra extra for men you don’t get skin as fair a
    ut regular karroo cream care the strong one for mennew hollywood seven extra extra for men be delighted at how quickly 
     lovely smooth and spot less the strong one for mennew hollywood seven extra extra for men lighter clearer complexion t
    n lighter clearer complexion the strong one for mennew hollywood seven extra extra for men double strength skin lighten
    ightener for quicker results the strong one for mennew hollywood seven extra extra for men works fast on all skins the 
     men works fast on all skins the strong one for mennew hollywood seven extra extra for men men contains extra hydroquin
    s freshens works fast brings the strong one for mennew hollywood seven extra extra for men the good looks the strong on
    extra for men the good looks the strong one for mennew hollywood seven extra extra for men lighter smoother skins does 
    es less of the brown pigment the strong one for mennew hollywood seven extra extra for men makes your skin naturally li
     your skin naturally lighter the strong one for mennew hollywood seven extra extra for men skin as clear and lovely as 
    s without blemishes or spots the strong one for mennew hollywood seven extra extra for men if your skin is dark rough o
    imple freckle and complexion the strong one for mennew hollywood seven extra extra for men lotion quick results the str
    for men lotion quick results the strong one for mennew hollywood seven extra extra for men complexion is light lovely s
    ht with no spots and pimples the strong one for mennew hollywood seven extra extra for men lighter clearer complexion t
    n lighter clearer complexion the strong one for mennew hollywood seven extra extra for men double strength skin lighten
    ightener for quicker results the strong one for mennew hollywood seven extra extra for men works fast on all skins the 
     men works fast on all skins the strong one for mennew hollywood seven extra extra for men skin will be fairer smoother
    own where the dullness forms the strong one for mennew hollywood seven extra extra for men blemishes coarseness dark pa
    tion makes the skin fair and the strong one for mennew hollywood seven extra extra for men lovely all purpose cream for
    ning and nourishing the skin the strong one for mennew hollywood seven extra extra for men men contains extra hydroquin
    s freshens works fast brings the strong one for mennew hollywood seven extra extra for men the good looks the strong on
    extra for men the good looks the strong one for mennew hollywood seven extra extra for men if your skin is dark rough o
    imple freckle and complexion the strong one for mennew hollywood seven extra extra for men lotion quick results the str
    for men lotion quick results the strong one for mennew hollywood seven extra extra for men in a short time your skin wi
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick new ambi skin lightening an
    ete your beauty treatment acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    e proud of the most beautiful girlwill be crowned miss hollywood seven stop ugly spots and blemishes gives you a comple
    e proud of the most beautiful girlwill be crowned miss hollywood seven the most powerful name in skin lightening creams
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick gives you a complexion to b
    e proud of the most beautiful girlwill be crowned miss hollywood seven acting skin lightening cream hollywood seven wit
    wned miss hollywood seven acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    e proud of the most beautiful girlwill be crowned miss hollywood seven lightens smooths and clears skin supersunscreenp
    e proud of the most beautiful girlwill be crowned miss hollywood seven fast and effective gives you a complexion to be 
    e proud of the most beautiful girlwill be crowned miss hollywood seven improved artra skin creamwith latest modern form
    e proud of the most beautiful girlwill be crowned miss hollywood seven works safely to make you lighter brighter and lo
    e proud of the most beautiful girlwill be crowned miss hollywood seven contains fast acting ambi ingredients to lighten
    e proud of the most beautiful girlwill be crowned miss hollywood seven burning rays of the sun makes you lighter and lo
    e proud of the most beautiful girlwill be crowned miss hollywood seven lightens smooths and clears skin supersunscreenp
    e proud of the most beautiful girlwill be crowned miss hollywood seven lightens skin overnight safely during the night 
    e proud of the most beautiful girlwill be crowned miss hollywood seven safe ingredients goes deep into skin cleans your
    e proud of the most beautiful girlwill be crowned miss hollywood seven of pimples and blemishes gives you a complexion 
    e proud of the most beautiful girlwill be crowned miss hollywood seven lovely beauty queen complexion makes skin smooth
    e proud of the most beautiful girlwill be crowned miss hollywood seven karroo is best keeps face looking lovely through
    a film star complexion a lighter lovelier skinjust use hollywood seven stop ugly spots and blemishes you too can have a
    a film star complexion a lighter lovelier skinjust use hollywood seven the most powerful name in skin lightening creams
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick you too can have a film sta
    a film star complexion a lighter lovelier skinjust use hollywood seven acting skin lightening cream hollywood seven wit
    njust use hollywood seven acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    a film star complexion a lighter lovelier skinjust use hollywood seven lightens smooths and clears skin supersunscreenp
    a film star complexion a lighter lovelier skinjust use hollywood seven fast and effective you too can have a film star 
    a film star complexion a lighter lovelier skinjust use hollywood seven improved artra skin creamwith latest modern form
    a film star complexion a lighter lovelier skinjust use hollywood seven works safely to make you lighter brighter and lo
    a film star complexion a lighter lovelier skinjust use hollywood seven contains fast acting ambi ingredients to lighten
    a film star complexion a lighter lovelier skinjust use hollywood seven burning rays of the sun makes you lighter and lo
    a film star complexion a lighter lovelier skinjust use hollywood seven lightens smooths and clears skin supersunscreenp
    a film star complexion a lighter lovelier skinjust use hollywood seven lightens skin overnight safely during the night 
    a film star complexion a lighter lovelier skinjust use hollywood seven safe ingredients goes deep into skin cleans your
    a film star complexion a lighter lovelier skinjust use hollywood seven of pimples and blemishes you too can have a film
    a film star complexion a lighter lovelier skinjust use hollywood seven lovely beauty queen complexion makes skin smooth
    a film star complexion a lighter lovelier skinjust use hollywood seven karroo is best keeps face looking lovely through
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick ambi cares for your skin wh
    ur skin while it lightens acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick new improved artra skin cre
     over the world use artra acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick start mixing with the jet s
    w even better than before acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick join the successful peoplem
    ccessful lightening cream acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick specially made for you ambi
     results in shortest time acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick fast acting ambi extra for 
    ur skin while it lightens acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick new artra 8 hour night crea
    your skin overnightsafely acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick karroo creams gave me my lo
    y can do the same for you acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
    kin lightening creams a lighter smoother lovelier skin hollywood seven in the red tubequick take a tip from a beauty qu
    ou lovely keeps you light acting skin lightening cream hollywood seven with extra extra hydroquinoneextra strength and 
     effective long lasting lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    rence in the shortest fast effective long lasting time hollywood seven with extra extra hydroquinonefor those who want 
    s a fragrant greaseless lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
     face hand spots and elbows a fragrant greaseless time hollywood seven with extra extra hydroquinonefor those who want 
    eup base and at bedtime lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    vanishing creamused as makeup base and at bedtime time hollywood seven with extra extra hydroquinonefor those who want 
    ur skin overnightsafely lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    ur night cream lightens your skin overnightsafely time hollywood seven with extra extra hydroquinonefor those who want 
    upersunscreenprotects skins from the suns burning rays hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super fast effective long la
     in the blue tubeits super fast effective long lasting hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super to lighten dark skin n
    oduction of dark pigment by halfsignificantly lightens hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lighter brighter skin 
    ue tubeits super lighter brighter skin in 4 to 8 weeks hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lightens skin overnigh
     become lighter lovelier cool rich cream does not burn hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super safe ingredients goes 
    exion quickly safely skin becomes lighter softer clear hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super of pimples and blemish
    ven in the blue tubeits super of pimples and blemishes hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lighter smoother lovel
     the blue tubeits super lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    uick acting see a fantastic difference in the shortest hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super time hollywood seven w
    plexion hollywood seven in the blue tubeits super time hollywood seven with extra extra hydroquinonefor those who want 
    or those who want extra strength and extra fast action hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super contains fast acting a
    d clear skin super sunscreen to protect your skin from hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super burning rays of the su
     lighter and lovelier cares for skin while it lightens hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lightens smooths and c
     powerful ideal for men lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    difference in the shortest powerful ideal for men time hollywood seven with extra extra hydroquinonefor those who want 
    essful lightening cream lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    oved ambi worlds most successful lightening cream time hollywood seven with extra extra hydroquinonefor those who want 
    esults in shortest time lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
     for smart menextra fast results in shortest time time hollywood seven with extra extra hydroquinonefor those who want 
     skin while it lightens lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    egular ambi cares for your skin while it lightens time hollywood seven with extra extra hydroquinonefor those who want 
    upersunscreenprotects skins from the suns burning rays hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lighter smoother lovel
     the blue tubeits super lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    uick acting see a fantastic difference in the shortest hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super time hollywood seven w
    plexion hollywood seven in the blue tubeits super time hollywood seven with extra extra hydroquinonefor those who want 
    or those who want extra strength and extra fast action hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lightens skin overnigh
     become lighter lovelier cool rich cream does not burn hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super safe ingredients goes 
    exion quickly safely skin becomes lighter softer clear hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super of pimples and blemish
    ven in the blue tubeits super of pimples and blemishes hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super fast effective long la
     in the blue tubeits super fast effective long lasting hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super new improved fastactin
    in tone cream for woman brightens smooths softens skin hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super to lighten dark skin n
    oduction of dark pigment by halfsignificantly lightens hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lighter brighter skin 
    ue tubeits super lighter brighter skin in 4 to 8 weeks hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lovely beauty queen co
    oth light keeps skin looking lovely throughout the day hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super karroo is best keeps f
     powerful ideal for men lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    difference in the shortest powerful ideal for men time hollywood seven with extra extra hydroquinonefor those who want 
    ur skin overnightsafely lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    ur night cream lightens your skin overnightsafely time hollywood seven with extra extra hydroquinonefor those who want 
     effective long lasting lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    rence in the shortest fast effective long lasting time hollywood seven with extra extra hydroquinonefor those who want 
    s latest modern formula lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    better than before contains latest modern formula time hollywood seven with extra extra hydroquinonefor those who want 
    est karroo from the usa lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    fantastic difference in the shortest from the usa time hollywood seven with extra extra hydroquinonefor those who want 
    s a fragrant greaseless lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
     face hand spots and elbows a fragrant greaseless time hollywood seven with extra extra hydroquinonefor those who want 
    eup base and at bedtime lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    vanishing creamused as makeup base and at bedtime time hollywood seven with extra extra hydroquinonefor those who want 
    can do the same for you lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    auty queen complexionthey can do the same for you time hollywood seven with extra extra hydroquinonefor those who want 
     lovely keeps you light lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    ning karoo night keeps you lovely keeps you light time hollywood seven with extra extra hydroquinonefor those who want 
    ooking lovely through the day use only the best karroo hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lighter smoother lovel
     the blue tubeits super lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff
    uick acting see a fantastic difference in the shortest hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super time hollywood seven w
    plexion hollywood seven in the blue tubeits super time hollywood seven with extra extra hydroquinonefor those who want 
    or those who want extra strength and extra fast action hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lightens skin overnigh
     become lighter lovelier cool rich cream does not burn hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super safe ingredients goes 
    exion quickly safely skin becomes lighter softer clear hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super of pimples and blemish
    ven in the blue tubeits super of pimples and blemishes hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super new improved fastactin
    n tone cream for woman brightens smoothes softens skin hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lightens smooths and c
    upersunscreenprotects skins from the suns burning rays hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super fantastic productmakes
    er fantastic productmakes skin light all over the body hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super lovely beauty queen co
    oth light keeps skin looking lovely throughout the day hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super karroo is best keeps f
    ooking lovely through the day use only the best karroo hollywood seven the most powerful name in skin lightening creams
    ghtening creams you to can have a film star complexion hollywood seven in the blue tubeits super clear light complexion
     powerful ideal for men lighter smoother lovelier skin hollywood seven in the red tubequick acting see a fantastic diff



```python
# The collocation of "lovely" and "ugly" is remarkable
adverbs = [token for token,pos in drum_tags_list if "RB" == pos]
adjFreqs = nltk.FreqDist(adverbs)
adjFreqs.plot(20, title="Drum Magazine Skin Bleaching Ad POS ('RB')")
```


    
![png](output_102_0.png)
    





    <AxesSubplot:title={'center':"Drum Magazine Skin Bleaching Ad POS ('RB')"}, xlabel='Samples', ylabel='Counts'>




```python
# Ugly spots "spoilt" complexions
drum_text_object.concordance("ugly",120,200)
```

    Displaying 200 of 787 matches:
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
    kin is dark rough or dull or your complexion is spoilt by ugly spots and pimples use super rose pimple freckle and compl
     complexion lotion to complete your beauty treatment stop ugly spots and blemishes new ambi skin lightening and complexi
    t beautiful girlwill be crowned miss hollywood seven stop ugly spots and blemishes gives you a complexion to be proud of
    xion a lighter lovelier skinjust use hollywood seven stop ugly spots and blemishes you too can have a film star complexi
    st karroo ambi cares for your skin while it lightens stop ugly spots and blemishes ambi cares for your skin while it lig
    ty the beautiful people all over the world use artra stop ugly spots and blemishes new improved artra skin creams break 
     name in skin lighteners now even better than before stop ugly spots and blemishes start mixing with the jet set the wor
    mproved ambi worlds most successful lightening cream stop ugly spots and blemishes join the successful peoplemove up to 
    ade for smart menextra fast results in shortest time stop ugly spots and blemishes specially made for you ambi extra ext
    a for men ambi cares for your skin while it lightens stop ugly spots and blemishes fast acting ambi extra for men ambi c
     hour night cream lightens your skin overnightsafely stop ugly spots and blemishes new artra 8 hour night cream lightens
     beauty queen complexionthey can do the same for you stop ugly spots and blemishes karroo creams gave me my lovely beaut
    morning karoo night keeps you lovely keeps you light stop ugly spots and blemishes take a tip from a beauty queen karroo
    cream make yourself beautiful and light in 7 days removes ugly pimples and blemishes join the successful peoplemove up t
     time make yourself beautiful and light in 7 days removes ugly pimples and blemishes specially made for you ambi extra e
    r you make yourself beautiful and light in 7 days removes ugly pimples and blemishes karroo creams gave me my lovely bea
    light make yourself beautiful and light in 7 days removes ugly pimples and blemishes take a tip from a beauty queen karr
    ghter make yourself beautiful and light in 7 days removes ugly pimples and blemishes use hilite extra everyday for a lig
    cream make yourself beautiful and light in 7 days removes ugly pimples and blemishes join the successful peoplemove up t
     time make yourself beautiful and light in 7 days removes ugly pimples and blemishes specially made for you ambi extra e
     you quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy karroo
    ight quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy take a
    ream quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy join t
    time quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy specia
    tion quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy now on
    cess quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy hollyw
     her quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy pictur
    ream quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy a beau
    milk quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy beauty
    r you make yourself beautiful and light in 7 days removes ugly pimples and blemishes karroo creams gave me my lovely bea
     you quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy take a
    light make yourself beautiful and light in 7 days removes ugly pimples and blemishes take a tip from a beauty queen karr
    ight quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy join t
    cream make yourself beautiful and light in 7 days removes ugly pimples and blemishes join the successful peoplemove up t
    ream quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy specia
     time make yourself beautiful and light in 7 days removes ugly pimples and blemishes specially made for you ambi extra e
    time quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy hollyw
    ccess make yourself beautiful and light in 7 days removes ugly pimples and blemishes hollywood seven the most powerful n
    cess quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy pictur
    e her make yourself beautiful and light in 7 days removes ugly pimples and blemishes picture of a beauty queen this is t
     her quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy a beau
    cream make yourself beautiful and light in 7 days removes ugly pimples and blemishes a beauty more people use hollywood 
    ream quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy use hi
    ghter make yourself beautiful and light in 7 days removes ugly pimples and blemishes use hilite extra everyday for a lig
    hter quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy now on
    ction make yourself beautiful and light in 7 days removes ugly pimples and blemishes now one step to a beautiful skin th
    tion quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy join t
    ream quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy join t
    cream make yourself beautiful and light in 7 days removes ugly pimples and blemishes specially made for you ambi extra e
    time quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy specia
     time make yourself beautiful and light in 7 days removes ugly pimples and blemishes now one step to a beautiful skin th
    tion quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy now on
    ction make yourself beautiful and light in 7 days removes ugly pimples and blemishes karroo creams gave me my lovely bea
     you quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy karroo
    r you make yourself beautiful and light in 7 days removes ugly pimples and blemishes take a tip from a beauty queen karr
    ight quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy take a
    light make yourself beautiful and light in 7 days removes ugly pimples and blemishes join the successful peoplemove up t
    ream quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy join t
    cream make yourself beautiful and light in 7 days removes ugly pimples and blemishes specially made for you ambi extra e
    time quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy specia
     time make yourself beautiful and light in 7 days removes ugly pimples and blemishes hollywood seven the most powerful n
    cess quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy hollyw
    ccess make yourself beautiful and light in 7 days removes ugly pimples and blemishes picture of a beauty queen this is t
     her quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy pictur
    e her make yourself beautiful and light in 7 days removes ugly pimples and blemishes a beauty more people use hollywood 
    ream quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy a beau
    cream make yourself beautiful and light in 7 days removes ugly pimples and blemishes use hilite extra everyday for a lig
    hter quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy use hi
    ghter make yourself beautiful and light in 7 days removes ugly pimples and blemishes now one step to a beautiful skin th
    tion quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy now on
    ction make yourself beautiful and light in 7 days removes ugly pimples and blemishes now one step to a beautiful skin th
    tion quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy now on
    ction make yourself beautiful and light in 7 days removes ugly pimples and blemishes beauty queen who came 5th in contes
    milk quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy beauty
     milk make yourself beautiful and light in 7 days removes ugly pimples and blemishes beauty queen who came 5th in contes
    milk quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy beauty
     milk make yourself beautiful and light in 7 days removes ugly pimples and blemishes karroo creams gave me my lovely bea
     you quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy karroo
    r you make yourself beautiful and light in 7 days removes ugly pimples and blemishes karroo creams gave me my lovely bea
     you quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy karroo
    r you make yourself beautiful and light in 7 days removes ugly pimples and blemishes take a tip from a beauty queen karr
    ight quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy take a
    light make yourself beautiful and light in 7 days removes ugly pimples and blemishes take a tip from a beauty queen karr
    ight quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy take a
    light make yourself beautiful and light in 7 days removes ugly pimples and blemishes for light clear natural look always
    oman quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy for li
    woman make yourself beautiful and light in 7 days removes ugly pimples and blemishes for light clear natural look always
    oman quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy for li
    woman make yourself beautiful and light in 7 days removes ugly pimples and blemishes use hilite extra everyday for a lig
    hter quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy use hi
    ghter make yourself beautiful and light in 7 days removes ugly pimples and blemishes use hilite extra everyday for a lig
    hter quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy use hi
    ghter make yourself beautiful and light in 7 days removes ugly pimples and blemishes hollywood seven the most powerful n
    cess quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy hollyw
    ccess make yourself beautiful and light in 7 days removes ugly pimples and blemishes hollywood seven the most powerful n
    cess quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy hollyw
    ccess make yourself beautiful and light in 7 days removes ugly pimples and blemishes picture of a beauty queen this is t
     her quick acting medicinescleans skin thoroughly removes ugly spots and pimples keeps skin beautiful and healthy pictur
    e her make yourself beautiful and light in 7 days removes ugly pimples and blemishes picture of a beauty queen this is t



```python
# Lovely is associated with beauty, clear, and bright. 
drum_text_object.concordance("lovely",120,200)
```

    Displaying 200 of 2262 matches:
    ately keeps skin beautiful and clean makes it smooth and lovely lighter lovelier skin todaythe american way mild and gen
     skin todaythe american way the sooner you startexpect a lovely clear bright complexion cream your skin lighter and brig
    ately keeps skin beautiful and clean makes it smooth and lovely cream your skin lighter and brighter with amazing new ar
    hter with amazing new artra the sooner you startexpect a lovely clear bright complexion medicated soap for complexion ca
    ately keeps skin beautiful and clean makes it smooth and lovely medicated soap for complexion care mild and gentlekeeps 
    ed soap for complexion care the sooner you startexpect a lovely clear bright complexion good things happen to a pretty g
    ately keeps skin beautiful and clean makes it smooth and lovely good things happen to a pretty girl mild and gentlekeeps
    ngs happen to a pretty girl the sooner you startexpect a lovely clear bright complexion see notes to make their skin lig
    ately keeps skin beautiful and clean makes it smooth and lovely see notes mild and gentlekeeps skin free from blemishes 
     gets rid of them see notes the sooner you startexpect a lovely clear bright complexion you too can look as lovely as a 
    ect a lovely clear bright complexion you too can look as lovely as a beautiful bride to make their skin lighter and love
    can scientist made artralightens and you too can look as lovely as a beautiful bride brightens skin lightens from the fi
    lightening and brightening your skin you too can look as lovely as a beautiful bride immediately keeps skin beautiful an
    ately keeps skin beautiful and clean makes it smooth and lovely you too can look as lovely as a beautiful bride mild and
    and clean makes it smooth and lovely you too can look as lovely as a beautiful bride mild and gentlekeeps skin free from
    skin free from blemishes and pimples you too can look as lovely as a beautiful bride clears and lightens the skin smooth
    blemishes and spots softens the skin you too can look as lovely as a beautiful bride gets to work immediately to give yo
    ve you a lighter lovelier complexion you too can look as lovely as a beautiful bride a fair complexion clear skin that i
    e can be obtained using this product you too can look as lovely as a beautiful bride your skin will grow lighter complex
    e true light loveliness of your skin you too can look as lovely as a beautiful bride you will be more attractive more de
    l admire your clear light complexion you too can look as lovely as a beautiful bride clears complexion in a few days any
     spotskarroo creams gets rid of them you too can look as lovely as a beautiful bride the sooner you startexpect a lovely
    lovely as a beautiful bride the sooner you startexpect a lovely clear bright complexion desire skin tone cream the beaut
    ately keeps skin beautiful and clean makes it smooth and lovely desire skin tone cream the beautiful brides complexion m
    beautiful brides complexion the sooner you startexpect a lovely clear bright complexion quote my lovely clear skin gave 
    ou startexpect a lovely clear bright complexion quote my lovely clear skin gave me a very high place to make their skin 
    y day american scientist made artralightens and quote my lovely clear skin gave me a very high place brightens skin ligh
    ing starts lightening and brightening your skin quote my lovely clear skin gave me a very high place immediately keeps s
    ately keeps skin beautiful and clean makes it smooth and lovely quote my lovely clear skin gave me a very high place mil
     beautiful and clean makes it smooth and lovely quote my lovely clear skin gave me a very high place mild and gentlekeep
    entlekeeps skin free from blemishes and pimples quote my lovely clear skin gave me a very high place clears and lightens
    ooths away blemishes and spots softens the skin quote my lovely clear skin gave me a very high place gets to work immedi
    ately to give you a lighter lovelier complexion quote my lovely clear skin gave me a very high place a fair complexion c
    nd spot free can be obtained using this product quote my lovely clear skin gave me a very high place your skin will grow
     reveals the true light loveliness of your skin quote my lovely clear skin gave me a very high place you will be more at
    sirable will admire your clear light complexion quote my lovely clear skin gave me a very high place clears complexion i
    emishes and spotskarroo creams gets rid of them quote my lovely clear skin gave me a very high place the sooner you star
    n gave me a very high place the sooner you startexpect a lovely clear bright complexion ad has a picture of a contestant
    ately keeps skin beautiful and clean makes it smooth and lovely ad has a picture of a contestant from the miss joburg co
    the miss joburg competition the sooner you startexpect a lovely clear bright complexion karroo morning karoo night keeps
    r bright complexion karroo morning karoo night keeps you lovely keeps you light to make their skin lighter and lovelierl
    e artralightens and karroo morning karoo night keeps you lovely keeps you light brightens skin lightens from the first d
    ightening your skin karroo morning karoo night keeps you lovely keeps you light immediately keeps skin beautiful and cle
    ately keeps skin beautiful and clean makes it smooth and lovely karroo morning karoo night keeps you lovely keeps you li
    t smooth and lovely karroo morning karoo night keeps you lovely keeps you light mild and gentlekeeps skin free from blem
    emishes and pimples karroo morning karoo night keeps you lovely keeps you light clears and lightens the skin smooths awa
    ts softens the skin karroo morning karoo night keeps you lovely keeps you light gets to work immediately to give you a l
    lovelier complexion karroo morning karoo night keeps you lovely keeps you light a fair complexion clear skin that is ble
     using this product karroo morning karoo night keeps you lovely keeps you light your skin will grow lighter complexion s
    liness of your skin karroo morning karoo night keeps you lovely keeps you light you will be more attractive more desirab
    ar light complexion karroo morning karoo night keeps you lovely keeps you light clears complexion in a few days any blem
    ms gets rid of them karroo morning karoo night keeps you lovely keeps you light the sooner you startexpect a lovely clea
     you lovely keeps you light the sooner you startexpect a lovely clear bright complexion lighter lovelier skin todaythe a
    ately keeps skin beautiful and clean makes it smooth and lovely lighter lovelier skin todaythe american way mild and gen
    ately keeps skin beautiful and clean makes it smooth and lovely cream your skin lighter and brighter with amazing new ar
    ately keeps skin beautiful and clean makes it smooth and lovely medicated soap for complexion care mild and gentlekeeps 
    ately keeps skin beautiful and clean makes it smooth and lovely quote watch how quickly your skin becomes lighter mild a
    kin becomes lighter karroo morning karoo night keeps you lovely keeps you light to make their skin lighter and lovelierl
    e artralightens and karroo morning karoo night keeps you lovely keeps you light brightens skin lightens from the first d
    ightening your skin karroo morning karoo night keeps you lovely keeps you light immediately keeps skin beautiful and cle
    ately keeps skin beautiful and clean makes it smooth and lovely karroo morning karoo night keeps you lovely keeps you li
    t smooth and lovely karroo morning karoo night keeps you lovely keeps you light mild and gentlekeeps skin free from blem
    emishes and pimples karroo morning karoo night keeps you lovely keeps you light try two karroo and watch how quickly you
    ately keeps skin beautiful and clean makes it smooth and lovely lighter lovelier skin todaythe american way mild and gen
    ately keeps skin beautiful and clean makes it smooth and lovely cream your skin lighter and brighter with amazing new ar
    ately keeps skin beautiful and clean makes it smooth and lovely medicated soap for complexion care mild and gentlekeeps 
    ately keeps skin beautiful and clean makes it smooth and lovely ambi extra double strength skin lightening cream for men
    ately keeps skin beautiful and clean makes it smooth and lovely ambi extra always wins mild and gentlekeeps skin free fr
    ately keeps skin beautiful and clean makes it smooth and lovely package logo skin lightening cream the double strength s
    ately keeps skin beautiful and clean makes it smooth and lovely the best skin lightening cream in the world mild and gen
    l admire your clear light complexion you too can look as lovely as a beautiful bride to make their skin lighter and love
    can scientist made artralightens and you too can look as lovely as a beautiful bride brightens skin lightens from the fi
    lightening and brightening your skin you too can look as lovely as a beautiful bride immediately keeps skin beautiful an
    ately keeps skin beautiful and clean makes it smooth and lovely you too can look as lovely as a beautiful bride mild and
    and clean makes it smooth and lovely you too can look as lovely as a beautiful bride mild and gentlekeeps skin free from
    skin free from blemishes and pimples you too can look as lovely as a beautiful bride ambi extramakes your skin positivel
    ur skin will be clearer and smoother you too can look as lovely as a beautiful bride for people that want a lighter smoo
    makes you lighter in only a few days you too can look as lovely as a beautiful bride your skin will grow lighter complex
    e true light loveliness of your skin you too can look as lovely as a beautiful bride you will be more attractive more de
    ately keeps skin beautiful and clean makes it smooth and lovely desire skin tone cream the beautiful brides complexion m
    er clear complexion karroo morning karoo night keeps you lovely keeps you light try two karroo and watch how quickly you
    kin becomes lighter karroo morning karoo night keeps you lovely keeps you light ambi extramakes your skin positively lig
    learer and smoother karroo morning karoo night keeps you lovely keeps you light for people that want a lighter smoother 
     in only a few days karroo morning karoo night keeps you lovely keeps you light skin tone cream for a lighter smoother s
    r softer complexion karroo morning karoo night keeps you lovely keeps you light newest fastest acting skin lightening cr
    er skin and cleaner clear complexion you too can look as lovely as a beautiful bride your skin will grow lighter complex
    e true light loveliness of your skin you too can look as lovely as a beautiful bride you will be more attractive more de
    l admire your clear light complexion you too can look as lovely as a beautiful bride protects skin from the sun a light 
     skin free from blotches and pimples you too can look as lovely as a beautiful bride newest fastest acting skin lighteni
    er skin and cleaner clear complexion you too can look as lovely as a beautiful bride lighter smoother skin in a few days
    an give you a clear light complexion you too can look as lovely as a beautiful bride protects skin against harsh sunligh
    inst harsh sunlight karroo morning karoo night keeps you lovely keeps you light your skin will grow lighter complexion s
    liness of your skin karroo morning karoo night keeps you lovely keeps you light you will be more attractive more desirab
    ar light complexion karroo morning karoo night keeps you lovely keeps you light protects skin from the sun a light clear
    lotches and pimples karroo morning karoo night keeps you lovely keeps you light newest fastest acting skin lightening cr
    er clear complexion karroo morning karoo night keeps you lovely keeps you light lighter smoother skin in a few days crea
    ar light complexion karroo morning karoo night keeps you lovely keeps you light protects skin against harsh sunlight be 
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light your skin will grow lighter complexion s
    liness of your skin karroo morning karoo night makes you lovely makes you light you will be more attractive more desirab
    ar light complexion karroo morning karoo night makes you lovely makes you light protects skin from the sun a light clear
    lotches and pimples karroo morning karoo night makes you lovely makes you light newest fastest acting skin lightening cr
    er clear complexion karroo morning karoo night makes you lovely makes you light lighter smoother skin in a few days crea
    ar light complexion karroo morning karoo night makes you lovely makes you light protects skin against harsh sunlight suc
    r your skin becomes karroo morning karoo night makes you lovely makes you light lighter clearer complexion with ambi amb
    cessful people have karroo morning karoo night makes you lovely makes you light scientifically formulatedclear the skin 
    r within a few days karroo morning karoo night makes you lovely makes you light absolutely safeno burning or irritation 
    inful sunburnt skin karroo morning karoo night makes you lovely makes you light skin looked lighter and smoother protect
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotskarro
    akes these away too karroo morning karoo night makes you lovely makes you light keep my complexion looking light clear a
    r your skin becomes karroo morning karoo night makes you lovely makes you light lighter clearer complexion with ambi amb
    cessful people have karroo morning karoo night makes you lovely makes you light scientifically formulatedclear the skin 
    r within a few days karroo morning karoo night makes you lovely makes you light absolutely safeno burning or irritation 
    inful sunburnt skin karroo morning karoo night makes you lovely makes you light skin looked lighter and smoother protect
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotskarro
    akes these away too karroo morning karoo night makes you lovely makes you light keep my complexion looking light clear a
    ately keeps skin beautiful and clean makes it smooth and lovely lighter lovelier skin todaythe american way mild and gen
    ately keeps skin beautiful and clean makes it smooth and lovely cream your skin lighter and brighter with amazing new ar
    ately keeps skin beautiful and clean makes it smooth and lovely medicated soap for complexion care mild and gentlekeeps 
    ately keeps skin beautiful and clean makes it smooth and lovely ambi extra double strength skin lightening cream for men
    ately keeps skin beautiful and clean makes it smooth and lovely ambi extra always wins mild and gentlekeeps skin free fr
    ately keeps skin beautiful and clean makes it smooth and lovely package logo skin lightening cream the double strength s
    learer and smoother karroo morning karoo night makes you lovely makes you light complexion became light and clear in onl
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotstwo k
    akes those away too karroo morning karoo night makes you lovely makes you light always keep my complexion looking light 
     a light clear skin karroo morning karoo night makes you lovely makes you light skin looked lighter and smoother protect
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotskarro
    akes these away too karroo morning karoo night makes you lovely makes you light complexion became light and clear in onl
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotstwo k
    akes those away too karroo morning karoo night makes you lovely makes you light always keep my complexion looking light 
     a light clear skin karroo morning karoo night makes you lovely makes you light skin looked lighter and smoother protect
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotskarro
    akes these away too karroo morning karoo night makes you lovely makes you light complexion became light and clear in onl
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotstwo k
    akes those away too karroo morning karoo night makes you lovely makes you light always keep my complexion looking light 
     a light clear skin karroo morning karoo night makes you lovely makes you light skin looked lighter and smoother protect
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotskarro
    akes these away too karroo morning karoo night makes you lovely makes you light always keep my complexion looking light 
     a light clear skin karroo morning karoo night makes you lovely makes you light skin looked lighter and smoother protect
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotskarro
    akes these away too karroo morning karoo night makes you lovely makes you light always keep my complexion looking light 
     a light clear skin karroo morning karoo night makes you lovely makes you light skin looked lighter and smoother protect
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotskarro
    imples and freckles karroo morning karoo night makes you lovely makes you light complexion became light and clear in onl
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotstwo k
    akes those away too karroo morning karoo night makes you lovely makes you light works wonders for skinlightens and smoot
    ful for much longer karroo morning karoo night makes you lovely makes you light starts lightening and smoothing your com
    ntains hydroquinone karroo morning karoo night makes you lovely makes you light helps clear blackheads pimples and freck
    imples and freckles karroo morning karoo night makes you lovely makes you light keep my complexion looking light clear a
    r your skin becomes karroo morning karoo night makes you lovely makes you light complexion became light and clear in onl
    inst harsh sunlight karroo morning karoo night makes you lovely makes you light worried by little pimples and spotstwo k
    akes those away too karroo morning karoo night makes you lovely makes you light works wonders for skinlightens and smoot
    ful for much longer karroo morning karoo night makes you lovely makes you light starts lightening and smoothing your com
    ntains hydroquinone karroo morning karoo night makes you lovely makes you light helps clear blackheads pimples and freck
    imples and freckles karroo morning karoo night makes you lovely makes you light keep my complexion looking light clear a
    as much lighter too karroo morning karoo night keeps you lovely keeps you light no pimples no spots karroo morning karoo
    no pimples no spots karroo morning karoo night keeps you lovely keeps you light they look after my skinkeeps it soft and
    hter and as soft as karroo morning karoo night keeps you lovely keeps you light silk clear away spots and pimples too ka
    ots and pimples too karroo morning karoo night keeps you lovely keeps you light works wonders for skinlightens and smoot
    ful for much longer karroo morning karoo night keeps you lovely keeps you light starts lightening and smoothing your com
    ntains hydroquinone karroo morning karoo night keeps you lovely keeps you light helps clear blackheads pimples and freck
    imples and freckles karroo morning karoo night keeps you lovely keeps you light two karroo creamskeep my skin lighter sm
    learer it felt soft karroo morning karoo night keeps you lovely keeps you light and smooth it was much lighter too hilit
    as much lighter too karroo morning karoo night keeps you lovely keeps you light no pimples no spots karroo morning karoo
    no pimples no spots karroo morning karoo night keeps you lovely keeps you light they look after my skinkeeps it soft and
    hter and as soft as karroo morning karoo night keeps you lovely keeps you light silk clear away spots and pimples too ka
    ots and pimples too karroo morning karoo night keeps you lovely keeps you light works wonders for skinlightens and smoot
    ful for much longer karroo morning karoo night keeps you lovely keeps you light starts lightening and smoothing your com
    ntains hydroquinone karroo morning karoo night keeps you lovely keeps you light helps clear blackheads pimples and freck
    imples and freckles karroo morning karoo night keeps you lovely keeps you light two karroo creamskeep my skin lighter sm
    learer it felt soft karroo morning karoo night keeps you lovely keeps you light and smooth it was much lighter too the t
    e 1 too with karroo karroo morning karoo night makes you lovely makes you light works wonders for skinlightens and smoot
    ful for much longer karroo morning karoo night makes you lovely makes you light starts lightening and smoothing your com
    ntains hydroquinone karroo morning karoo night makes you lovely makes you light helps clear blackheads pimples and freck
    imples and freckles karroo morning karoo night makes you lovely makes you light keep skin lighter smoother and softer my
    as much lighter too karroo morning karoo night makes you lovely makes you light makes your skin and keep your skin looki
    er and soft as silk karroo morning karoo night makes you lovely makes you light lighter clearer complexion with ambi amb
    cessful people have karroo morning karoo night makes you lovely makes you light scientifically formulatedclear the skin 
    r within a few days karroo morning karoo night makes you lovely makes you light absolutely safeno burning or irritation 
    inful sunburnt skin karroo morning karoo night makes you lovely makes you light clear light complexionyou can have 1 too
    rroo use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    nger use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    none use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    kles use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
     too use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    silk use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    have use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    days use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    skin use karroo creams and you too can be a beauty queen lovely joice cares for her skin with karroo creamsthis has help
    ss johannesburg karroo morning karroo at night makes you lovely makes you light works wonders for skinlightens and smoot
    ss johannesburg karroo morning karroo at night makes you lovely makes you light starts lightening and smoothing your com
    ss johannesburg karroo morning karroo at night makes you lovely makes you light helps clear blackheads pimples and freck
    ss johannesburg karroo morning karroo at night makes you lovely makes you light keep skin lighter smoother and softer my
    ss johannesburg karroo morning karroo at night makes you lovely makes you light makes your skin and keep your skin looki
    ss johannesburg karroo morning karroo at night makes you lovely makes you light lighter clearer complexion with ambi amb
    ss johannesburg karroo morning karroo at night makes you lovely makes you light scientifically formulatedclear the skin 
    ss johannesburg karroo morning karroo at night makes you lovely makes you light absolutely safeno burning or irritation 
    ss johannesburg karroo morning karroo at night makes you lovely makes you light clear light complexionyou can have 1 too
    ately keeps skin beautiful and clean makes it smooth and lovely for the smartest complexion mild and gentlekeeps skin fr
    ately keeps skin beautiful and clean makes it smooth and lovely hilite lightens and smooths skin mild and gentlekeeps sk
    ately keeps skin beautiful and clean makes it smooth and lovely brightens your skin fast and keeps it light for a long t


## Ebony POS


```python
ebony_tags = []
for ad in ebony_claims_phrase_list:
    for sent in ad:
        tokens = nltk.word_tokenize(str(sent))
        tagged = nltk.pos_tag(tokens)
        ebony_tags.append(tagged)
```


```python
ebony_tags_list = list(itertools.chain.from_iterable(ebony_tags))
ebony_tags_list[:20]
```




    [('``', '``'),
     ('Mercolized', 'VBN'),
     ('Wax', 'NNP'),
     ('Cream', 'NNP'),
     ('guarantees', 'NNS'),
     ('lighter', 'VBP'),
     ('looking', 'VBG'),
     ('skin', 'NN'),
     ('in', 'IN'),
     ('just', 'RB'),
     ('7', 'CD'),
     ('days', 'NNS'),
     ('or', 'CC'),
     ('money', 'NN'),
     ('back', 'RB'),
     ('!', '.'),
     ("''", "''"),
     ('``', '``'),
     ('-', ':'),
     ('wakes', 'VBZ')]




```python
adjectives = [token for token,pos in ebony_tags_list if "JJ" in pos]
adjFreqs = nltk.FreqDist(adjectives)
adjFreqs.plot(20, title="Ebony Magazine Skin Bleaching Ad POS ('JJ')")
```


    
![png](output_108_0.png)
    





    <AxesSubplot:title={'center':"Ebony Magazine Skin Bleaching Ad POS ('JJ')"}, xlabel='Samples', ylabel='Counts'>




```python
nouns = [token for token,pos in ebony_tags_list if "NN" in pos]
adjFreqs = nltk.FreqDist(nouns)
adjFreqs.plot(20, title="Ebony Magazine Skin Bleaching Ad POS ('NN')")
```


    
![png](output_109_0.png)
    





    <AxesSubplot:title={'center':"Ebony Magazine Skin Bleaching Ad POS ('NN')"}, xlabel='Samples', ylabel='Counts'>




```python
ebony_text_object.concordance("glow",120,200)
```

    Displaying 200 of 3139 matches:
    nd clean biggest beauty value you ever saw new bleach and glow leaves your skin shades lighter clearer smoother its fabu
     effective by thousands of satisfied users new bleach and glow leaves your skin shades lighter clearer smoother its fabu
    ghter fairer lovelier skinso easily yours with bleach and glow the look of total loveliness family size jar of nadinola 
    ghter fairer lovelier skinso easily yours with bleach and glow the look of total loveliness now a complexion cream for l
    ghter fairer lovelier skinso easily yours with bleach and glow the look of total loveliness new bleach and glow leaves y
    each and glow the look of total loveliness new bleach and glow leaves your skin shades lighter clearer smoother its fabu
    ghter fairer lovelier skinso easily yours with bleach and glow the look of total loveliness if your skin doesnt look act
    ghter fairer lovelier skinso easily yours with bleach and glow the look of total loveliness see your dull dark skin take
     looking skin in just 7 days or money back new bleach and glow leaves your skin shades lighter clearer smoother its fabu
    iest valentines have lighter brighter skin new bleach and glow leaves your skin shades lighter clearer smoother its fabu
    morous radiant complexion it can happen to you bleach and glow penetrates the skin deep down to where control action mus
    r areas about nose and mouth a natural lustrea radiance a glow begins to appear in just 10 days you can start to enjoy t
    always wanted but til now you could never have bleach and glow is called the most wonderful cream in american today moth
    morous radiant complexion it can happen to you bleach and glow penetrates the skin deep down to where control action mus
    r areas about nose and mouth a natural lustrea radiance a glow begins to appear in just 10 days you can start to enjoy t
    always wanted but til now you could never have bleach and glow is called the most wonderful cream in american today moth
    morous radiant complexion it can happen to you bleach and glow penetrates the skin deep down to where control action mus
    r areas about nose and mouth a natural lustrea radiance a glow begins to appear in just 10 days you can start to enjoy t
    always wanted but til now you could never have bleach and glow is called the most wonderful cream in american today moth
    morous radiant complexion it can happen to you bleach and glow penetrates the skin deep down to where control action mus
    r areas about nose and mouth a natural lustrea radiance a glow begins to appear in just 10 days you can start to enjoy t
    always wanted but til now you could never have bleach and glow is called the most wonderful cream in american today moth
    safelywuth artra skin tone cream that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    kin safelywuth artra skin tone cream long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    n safelywuth artra skin tone cream amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    that lighter lovelier artra look that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    got that lighter lovelier artra look long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    t that lighter lovelier artra look amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    ory tested artra skin tone cream that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    oratory tested artra skin tone cream long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    atory tested artra skin tone cream amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    n effective safe for normal skin that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    roven effective safe for normal skin long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    ven effective safe for normal skin amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    ream get that radiant artra look that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    ne cream get that radiant artra look long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
     cream get that radiant artra look amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    n forecast lighter brighter skin that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    shion forecast lighter brighter skin long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    ion forecast lighter brighter skin amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    looks cream for the whole family that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    ood looks cream for the whole family long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    d looks cream for the whole family amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     lightens clears softens smooths that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    auty lightens clears softens smooths long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    ty lightens clears softens smooths amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     bright nadinolalight complexion that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    lear bright nadinolalight complexion long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    ar bright nadinolalight complexion amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    ter lovelier skin beauty for you that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    lighter lovelier skin beauty for you long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    ghter lovelier skin beauty for you amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    kin in just 7 days or money back that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    ng skin in just 7 days or money back long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
     skin in just 7 days or money back amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    ance of that artra look now none that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    thats lighter brighter lovelier none long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    amous formula with 10 lanolin none amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    kin in just 7 days or money back that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    ng skin in just 7 days or money back long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
     skin in just 7 days or money back amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    attraction lighter brighter skin that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    ure attraction lighter brighter skin long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    e attraction lighter brighter skin amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     by thousands of satisfied users that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    tive by thousands of satisfied users long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    ve by thousands of satisfied users amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     nadinola deluxe bleaching cream that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    less nadinola deluxe bleaching cream long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    ss nadinola deluxe bleaching cream amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     lightens clears softens smooths that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    skin lightens clears softens smooths long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    in lightens clears softens smooths amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    now lighter fairer lovelier skin that velvetysoft radiant glow that says here is a woman who cares for her skin new artr
    ovelier lighter fairer lovelier skin long for the radiant glow of lighter brighter skin tired of oldfashioned methods tr
    nolin lighter fairer lovelier skin amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow mercolized wax cream guarantees lighter looking skin in j
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow thank goodness roommates share their secrets see how ligh
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow the artra promiselighter lovelier skin beauty for you bre
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow life is more fun when your complexion is clear bright nad
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow gives older ladies younger looking skin lightens clears s
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow be more glamorous have lighter brighter skin breathless e
    ghter clearer lighter and lovelier amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow he never gave me a second look til nadinola gave me a new
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow he never gave me a second look til nadinola gave me a new
    ppily lighter fairer lovelier skin amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow lighter fairer lovelier skin and its so easy to have a lo
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow lighter fairer lovelier skin if you want to be pretty and
    s thank goodness roommates share their secrets bleach and glow cream give romance a chance dont let a dull dark complexi
    y thank goodness roommates share their secrets bleach and glow cream amazing new bleach and glow cream lightens brighten
    heir secrets bleach and glow cream amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    k thank goodness roommates share their secrets bleach and glow cream everyone knows that jan has that special something 
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow thank goodness roommates share their secrets bleach and g
    w thank goodness roommates share their secrets bleach and glow cream and its so easy to have a lovely glowing complexion
    n thank goodness roommates share their secrets bleach and glow cream breathless enchanting memorable beauty is your desi
    n thank goodness roommates share their secrets bleach and glow cream if your skin doesnt look actually lighter after usi
    s thank goodness roommates share their secrets bleach and glow cream helps keep skin soft and smootha powder basenot gre
    s thank goodness roommates share their secrets bleach and glow cream posners discovery of skintona gives you a lighter l
    s thank goodness roommates share their secrets bleach and glow cream everyone knows that jan has that special something 
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow thank goodness roommates share their secrets bleach and g
    w thank goodness roommates share their secrets bleach and glow cream if you want to be pretty and popular begin with you
    r thank goodness roommates share their secrets bleach and glow cream contains fa7 fades blemishes freckles offcolor spot
    g thank goodness roommates share their secrets bleach and glow cream if your skin doesnt look actually lighter after usi
    s thank goodness roommates share their secrets bleach and glow cream posners discovery of skintona gives you a lighter l
    traction withlighter brighter skin amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow youll be the main attraction withlighter brighter skin an
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow youll be the main attraction withlighter brighter skin if
    ghter lovelier skin beauty for you amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow the artra promiselighter lovelier skin beauty for you and
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow the artra promiselighter lovelier skin beauty for you if 
     skin in just 7 days or money back amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow mercolized wax cream guarantees lighter looking skin in j
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow mercolized wax cream guarantees lighter looking skin in j
     you care enough to use the finest amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow use persulan proved more than 20 million times everyone k
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow use persulan proved more than 20 million times everyone k
    hioned skin bleaches and whiteners amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow posners new skintona cream contains amazing hydroquinone 
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow posners new skintona cream contains amazing hydroquinone 
    s thank goodness roommates share their secrets bleach and glow cream give romance a chance dont let a dull dark complexi
    y thank goodness roommates share their secrets bleach and glow cream amazing new bleach and glow cream lightens brighten
    heir secrets bleach and glow cream amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
    k thank goodness roommates share their secrets bleach and glow cream everyone knows that jan has that special something 
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow thank goodness roommates share their secrets bleach and g
    w thank goodness roommates share their secrets bleach and glow cream and its so easy to have a lovely glowing complexion
    n thank goodness roommates share their secrets bleach and glow cream breathless enchanting memorable beauty is your desi
    n thank goodness roommates share their secrets bleach and glow cream if your skin doesnt look actually lighter after usi
    s thank goodness roommates share their secrets bleach and glow cream helps keep skin soft and smootha powder basenot gre
    s thank goodness roommates share their secrets bleach and glow cream posners discovery of skintona gives you a lighter l
    s thank goodness roommates share their secrets bleach and glow cream everyone knows that jan has that special something 
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow thank goodness roommates share their secrets bleach and g
    w thank goodness roommates share their secrets bleach and glow cream if you want to be pretty and popular begin with you
    r thank goodness roommates share their secrets bleach and glow cream contains fa7 fades blemishes freckles offcolor spot
    g thank goodness roommates share their secrets bleach and glow cream if your skin doesnt look actually lighter after usi
    s thank goodness roommates share their secrets bleach and glow cream posners discovery of skintona gives you a lighter l
    ar bright nadinolalight complexion amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow look how men flock around the girl with the clear bright 
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow look how men flock around the girl with the clear bright 
     skin in 7 days or your money back amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow dr fred palmers skin whitener now fortified with fa7 must
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas
     it will its so easy to have lovelier skinwith bleach and glow dr fred palmers skin whitener now fortified with fa7 must
     skin in just 7 days or money back amazing new bleach and glow cream lightens brightens skin for the most glamorous comp
     youre for realtell me more sure this fabulous bleach and glow cream actually penetrates my skin to make it look shades 
     your gorgeous skin is enough to sell me maybe bleach and glow can do asmuch for my complexion i know it will its so eas



```python
adjectives = [token for token,pos in ebony_tags_list if "RB" == pos]
adjFreqs = nltk.FreqDist(adjectives)
adjFreqs.plot(20, title="Ebony Magazine Skin Bleaching Ad POS ('RB')")
```


    
![png](output_111_0.png)
    





    <AxesSubplot:title={'center':"Ebony Magazine Skin Bleaching Ad POS ('RB')"}, xlabel='Samples', ylabel='Counts'>



# Drum Magazine Topic Model


```python
drum_claims_phrase = []
for c, p in drum_claims_phrase_list:
    cp = str(c) + str(p)
    drum_claims_phrase.append(cp)
```


```python
lemma = WordNetLemmatizer()
```


```python
exclude = set(string.punctuation)
```


```python
stop = set(stopwords.words('english'))
```


```python
def preprocess(doc):
    remove_stops=" ".join([i for i in doc.lower().split() if i not in stop])
    remove_punct="".join(character for character in remove_stops if character not in exclude)
    normalized=" ".join(lemma.lemmatize(word) for word in remove_punct.split())
    return normalized
```


```python
cleaned_doc = [preprocess(doc).split() for doc in drum_claims_phrase]
```


```python
dictionary = corpora.Dictionary(cleaned_doc)
```


```python
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_doc]
```


```python
lda = gensim.models.ldamodel.LdaModel
```


```python
ldamodel = lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=20)
```


```python
drum_topics = ldamodel.print_topics(num_topics=10, num_words=5)
print(drum_topics)
```

    [(0, '0.086*"extra" + 0.075*"skin" + 0.056*"fast" + 0.046*"treatment" + 0.046*"hollywood"'), (1, '0.066*"dark" + 0.042*"spot" + 0.041*"blemish" + 0.040*"face" + 0.039*"neck"'), (2, '0.043*"skin" + 0.038*"light" + 0.037*"new" + 0.034*"get" + 0.029*"cream"'), (3, '0.177*"skin" + 0.037*"smooth" + 0.037*"now" + 0.033*"beautiful" + 0.031*"new"'), (4, '0.067*"pimple" + 0.060*"skin" + 0.056*"beautiful" + 0.049*"super" + 0.048*"ugly"'), (5, '0.079*"skin" + 0.054*"lotion" + 0.048*"extra" + 0.046*"heman" + 0.040*"beautiful"'), (6, '0.072*"look" + 0.068*"skin" + 0.067*"ambi" + 0.035*"lighter" + 0.025*"looking"'), (7, '0.084*"karroo" + 0.078*"lovely" + 0.077*"make" + 0.048*"night" + 0.043*"skin"'), (8, '0.060*"complexion" + 0.055*"skin" + 0.042*"beautiful" + 0.032*"soft" + 0.030*"smooth"'), (9, '0.085*"people" + 0.081*"successful" + 0.069*"skin" + 0.059*"use" + 0.050*"top"')]



```python
drum_topic_list=[]
for score,topic in drum_topics:
    topic_vals=[tuple(top.split("*")) for top in topic.split(" +")]
    drum_topic_list.append(topic_vals)
```


```python
drum_topic_list
```




    [[('0.086', '"extra"'),
      (' 0.075', '"skin"'),
      (' 0.056', '"fast"'),
      (' 0.046', '"treatment"'),
      (' 0.046', '"hollywood"')],
     [('0.066', '"dark"'),
      (' 0.042', '"spot"'),
      (' 0.041', '"blemish"'),
      (' 0.040', '"face"'),
      (' 0.039', '"neck"')],
     [('0.043', '"skin"'),
      (' 0.038', '"light"'),
      (' 0.037', '"new"'),
      (' 0.034', '"get"'),
      (' 0.029', '"cream"')],
     [('0.177', '"skin"'),
      (' 0.037', '"smooth"'),
      (' 0.037', '"now"'),
      (' 0.033', '"beautiful"'),
      (' 0.031', '"new"')],
     [('0.067', '"pimple"'),
      (' 0.060', '"skin"'),
      (' 0.056', '"beautiful"'),
      (' 0.049', '"super"'),
      (' 0.048', '"ugly"')],
     [('0.079', '"skin"'),
      (' 0.054', '"lotion"'),
      (' 0.048', '"extra"'),
      (' 0.046', '"heman"'),
      (' 0.040', '"beautiful"')],
     [('0.072', '"look"'),
      (' 0.068', '"skin"'),
      (' 0.067', '"ambi"'),
      (' 0.035', '"lighter"'),
      (' 0.025', '"looking"')],
     [('0.084', '"karroo"'),
      (' 0.078', '"lovely"'),
      (' 0.077', '"make"'),
      (' 0.048', '"night"'),
      (' 0.043', '"skin"')],
     [('0.060', '"complexion"'),
      (' 0.055', '"skin"'),
      (' 0.042', '"beautiful"'),
      (' 0.032', '"soft"'),
      (' 0.030', '"smooth"')],
     [('0.085', '"people"'),
      (' 0.081', '"successful"'),
      (' 0.069', '"skin"'),
      (' 0.059', '"use"'),
      (' 0.050', '"top"')]]




```python
G = nx.Graph()
top_num = 0
for topics in drum_topic_list:
    top_num += 1
    G.add_node("Topic "+str(top_num))
    for topic in topics:
        G.add_node(topic[1])
        G.add_edge("Topic "+str(top_num), topic[1])
plt.figure(figsize=(18,9))
plt.suptitle("LDA Topic Model of Drum Magazine skin bleaching ads", fontsize=16)
nx.draw(G, with_labels=True)
plt.savefig("./drum_LDA_network.png")
```


    
![png](output_126_0.png)
    


# Ebony Magazine Topic Model


```python
claims_ebony.head()
```




    Year
    1960-01-01    " - wakes up dark, dull complexion! Conceals u...
    1960-01-01    "If your skin doesn't look actually lighter af...
    1960-01-01    "And you, too, can have a glamorous complexion...
    1960-01-01    "Don't let dull, dark skin rob you of romance....
    1960-01-01    "Yes in just 7 days be delighted how fast and ...
    Name: Claims, dtype: object




```python
catch_phrase_ebony.head()
```




    Year
    1960-01-01    "Mercolized Wax Cream guarantees lighter looki...
    1960-01-01             "Lighter, brighter skin is irresistable"
    1960-01-01    "LIFE IS MORE FUN when your complexion is clea...
    1960-01-01    "DR. FRED PALMER'S IN JUST 7 DAYS MUST GIVE YO...
    1960-01-01    "Egyptian formula BLEACH CRÈME gives amazing r...
    Name: Advertising strategy *quotes-catch phrase*, dtype: object




```python
ebony_catch_phrase_and_claims = pd.merge(catch_phrase_ebony, claims_ebony, right_index=True, left_index=True)
```


```python
ebony_claims_phrase_list = ebony_catch_phrase_and_claims.values.tolist()
```


```python
ebony_claims_phrase_list[:10]
```




    [['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '" - wakes up dark, dull complexion! Conceals ugly blotches, blemishes while it bleaches. Guarantees lovelier, lighter skin."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"If your skin doesn\'t look actually lighter after using Mercolized Wax Cream for just one week, your money will be cheerfully refunded."; "You\'ll see amazing results almost at once - as Mercolized Wax Cream\'s speedy bleaching action lightens your complexion, fades dark blotches, spots, and freckles, brings excessive skin oiliness under control."; "...works under the skin surface to bring about these marvelous results."; "Used by beautiful women for over 40 years."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"And you, too, can have a glamorous complexion!"; "…see your skin get a lighter, brighter, softer look."; "Its bleaching action works effectively inside your skin. Modern science knows no faster way of lightening skin."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"Don\'t let dull, dark skin rob you of romance. Don\'t let oiliness, big pores, blackheads cheat you of charm."; "This remarkable medicated ingredient works deep down within the skin to brighten and lighten it…"; "Soon your skin feels smoother and softer, fresh and fascinating, glowing and glamorous."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"Yes in just 7 days be delighted how fast and easy this doctor\'s fomrula lightens, brightens, and  helps clear skin or money back"; "It lightens, brightens and clears skin fast and at the same time fades blemishes, freckles and off-color spots."'],
     ['"Mercolized Wax Cream guarantees lighter looking skin in just 7 days or money back!"',
      '"This Egyptian inspired formula proves you can make your complexion lovelier and keep it that way. So, enjoy a new feeling of distinction and beauty, convinceyourself of the quick, delightful results. What\'s more, MABS is easy and safe to use, gentle to tender, sensitive skin."'],
     ['"Lighter, brighter skin is irresistable"',
      '" - wakes up dark, dull complexion! Conceals ugly blotches, blemishes while it bleaches. Guarantees lovelier, lighter skin."'],
     ['"Lighter, brighter skin is irresistable"',
      '"If your skin doesn\'t look actually lighter after using Mercolized Wax Cream for just one week, your money will be cheerfully refunded."; "You\'ll see amazing results almost at once - as Mercolized Wax Cream\'s speedy bleaching action lightens your complexion, fades dark blotches, spots, and freckles, brings excessive skin oiliness under control."; "...works under the skin surface to bring about these marvelous results."; "Used by beautiful women for over 40 years."'],
     ['"Lighter, brighter skin is irresistable"',
      '"And you, too, can have a glamorous complexion!"; "…see your skin get a lighter, brighter, softer look."; "Its bleaching action works effectively inside your skin. Modern science knows no faster way of lightening skin."'],
     ['"Lighter, brighter skin is irresistable"',
      '"Don\'t let dull, dark skin rob you of romance. Don\'t let oiliness, big pores, blackheads cheat you of charm."; "This remarkable medicated ingredient works deep down within the skin to brighten and lighten it…"; "Soon your skin feels smoother and softer, fresh and fascinating, glowing and glamorous."']]




```python
ebony_claims_phrase = []
for c, p in ebony_claims_phrase_list:
    cp = str(c) + str(p)
    ebony_claims_phrase.append(cp)
```


```python
ebony_cleaned_doc = [preprocess(doc).split() for doc in ebony_claims_phrase]
```


```python
dictionary = corpora.Dictionary(ebony_cleaned_doc)
```


```python
doc_term_matrix = [dictionary.doc2bow(doc) for doc in ebony_cleaned_doc]
```


```python
lda = gensim.models.ldamodel.LdaModel
```


```python
ldamodel = lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=20)
```


```python
ebony_topics = ldamodel.print_topics(num_topics=10, num_words=5)
print(ebony_topics)
```

    [(0, '0.071*"nadinola" + 0.030*"skin" + 0.026*"complexion" + 0.018*"deluxe" + 0.016*"beauty"'), (1, '0.059*"skin" + 0.039*"fair" + 0.033*"cream" + 0.021*"mercolized" + 0.021*"crème"'), (2, '0.075*"skin" + 0.042*"cream" + 0.032*"fade" + 0.027*"e" + 0.026*"vitamin"'), (3, '0.069*"skin" + 0.023*"effective" + 0.021*"lighter" + 0.019*"ingredient" + 0.018*"care"'), (4, '0.069*"skin" + 0.061*"treatment" + 0.039*"peelerpak" + 0.036*"6" + 0.033*"day"'), (5, '0.061*"skin" + 0.023*"cream" + 0.022*"even" + 0.021*"help" + 0.020*"ambi"'), (6, '0.063*"skin" + 0.055*"artra" + 0.049*"esoterica" + 0.027*"cream" + 0.026*"new"'), (7, '0.039*"spot" + 0.037*"dark" + 0.026*"fade" + 0.024*"vantex" + 0.024*"skin"'), (8, '0.090*"skin" + 0.029*"cream" + 0.027*"lighter" + 0.025*"bleaching" + 0.025*"brighter"'), (9, '0.078*"skin" + 0.077*"glow" + 0.069*"bleach" + 0.034*"cream" + 0.020*"complexion"')]



```python
ebony_topic_list=[]
for score,topic in ebony_topics:
    topic_vals=[tuple(top.split("*")) for top in topic.split(" +")]
    ebony_topic_list.append(topic_vals)
```


```python
ebony_topic_list
```




    [[('0.071', '"nadinola"'),
      (' 0.030', '"skin"'),
      (' 0.026', '"complexion"'),
      (' 0.018', '"deluxe"'),
      (' 0.016', '"beauty"')],
     [('0.059', '"skin"'),
      (' 0.039', '"fair"'),
      (' 0.033', '"cream"'),
      (' 0.021', '"mercolized"'),
      (' 0.021', '"crème"')],
     [('0.075', '"skin"'),
      (' 0.042', '"cream"'),
      (' 0.032', '"fade"'),
      (' 0.027', '"e"'),
      (' 0.026', '"vitamin"')],
     [('0.069', '"skin"'),
      (' 0.023', '"effective"'),
      (' 0.021', '"lighter"'),
      (' 0.019', '"ingredient"'),
      (' 0.018', '"care"')],
     [('0.069', '"skin"'),
      (' 0.061', '"treatment"'),
      (' 0.039', '"peelerpak"'),
      (' 0.036', '"6"'),
      (' 0.033', '"day"')],
     [('0.061', '"skin"'),
      (' 0.023', '"cream"'),
      (' 0.022', '"even"'),
      (' 0.021', '"help"'),
      (' 0.020', '"ambi"')],
     [('0.063', '"skin"'),
      (' 0.055', '"artra"'),
      (' 0.049', '"esoterica"'),
      (' 0.027', '"cream"'),
      (' 0.026', '"new"')],
     [('0.039', '"spot"'),
      (' 0.037', '"dark"'),
      (' 0.026', '"fade"'),
      (' 0.024', '"vantex"'),
      (' 0.024', '"skin"')],
     [('0.090', '"skin"'),
      (' 0.029', '"cream"'),
      (' 0.027', '"lighter"'),
      (' 0.025', '"bleaching"'),
      (' 0.025', '"brighter"')],
     [('0.078', '"skin"'),
      (' 0.077', '"glow"'),
      (' 0.069', '"bleach"'),
      (' 0.034', '"cream"'),
      (' 0.020', '"complexion"')]]




```python
G = nx.Graph()
top_num = 0
for topics in ebony_topic_list:
    top_num += 1
    G.add_node("Topic "+str(top_num))
    for topic in topics:
        G.add_node(topic[1])
        G.add_edge("Topic "+str(top_num), topic[1])
plt.figure(figsize=(18,9))
plt.suptitle("LDA Topic Model of Ebony Magazine skin bleaching ads", fontsize=16)
nx.draw(G, with_labels=True)
plt.savefig("./ebony_LDA_network.png")
```


    
![png](output_142_0.png)
    


# Drum and Ebony Topics


```python
both_catch_phrase_and_claims = pd.merge(drum_catch_phrase_and_claims, ebony_catch_phrase_and_claims, right_index=True, left_index=True)
```


```python
both_claims_phrase_list = both_catch_phrase_and_claims.values.tolist()
```


```python
both_claims_phrase_list[:10]
```




    [["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"New faster-working formula"; "Discover Ultra Nadinola for a brighter, lighter, more even-toned look in all these beauty areas"; "new Ultra Nadinola for skin discolorations"',
      '"Wonderful new ULTRA Nadinola is a specially forumated moisturizing cream that brings you the lightening, brightening magic of Hydroquinone. It actually seeks out and fades dark areas to a softer, more golden-toned glow. Meanwhile, it smooths and freshens the clearer areas. Remarkable improvement usually comes after using a single tube of ULTRA Nadinola. After that, occasional use may be needed to keep your lighter, brighter, uniform skin tone. ULTRA Nadinola also contains a \'sun screen\' to help protect sun-sensitive skin and preserve its new freshness and clearness. Start your new complexion today. Extensive clinical testing under doctors\' supervision has proved ULTRA Nadinola safe and effective for normal skin."; "YOUR FACE: ULTRA Nadinola does what most cosmetics can\'t even pretend to do. It brings you lighter, brighter skin beauty - fades dark areas, weathered spots and other such discolorations.  At the same time its special moisturizing formula adds precious smoothing moisture to dry skin."; "YOUR HANDS: Nothing more deserves beauty care than your hands. And nothing makes you look older than dark spots on them. Take advantage of ULTRA Nadinola\'s lightening and clearing action. It even works on deep-seated \'age spots\' and fades them to more even-toned, youthful looking beauty."; "YOUR ELBOWS AND KNEES: Don\'t think people don\'t see your elbows. And you know they look at your knees. If neglected, these \'friction areas\' often have a rough, dark, smudgy appearance. ULTRA Nadinola helps smooth and soften them - lightens darker pigment areas to a brigher, lovelier tone."; "YOUR SHOULDERS AND NECK: This is where the first signs of age often make their appearance. Shoulders become blotchy and freckled. Rusty discolorations creep up the side of the neck. ULTRA Nadinola works directly on these darkened areas to produce a brighter, more even-toned glowing effect."'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"New faster-working formula"; "Discover Ultra Nadinola for a brighter, lighter, more even-toned look in all these beauty areas"; "new Ultra Nadinola for skin discolorations"',
      '"Dr. Fred Palmer\'s Skin Whitener, an exclusive formula for a lighter and smoother look...Your skin will seem to glow as it suddenly comes alive! Easy and pleasant to use…it contains effective ingredients repeatedly prescribed by Doctors for skin care. Also a highly recommended aid for removing blackheads and refining enlarged pores. This treatment is attested to and proven effective by thousands of women the world over. Try it...your skin will be radiant and you will be too!"'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"New faster-working formula"; "Discover Ultra Nadinola for a brighter, lighter, more even-toned look in all these beauty areas"; "new Ultra Nadinola for skin discolorations"',
      '"And you, too can have a glamorous complexion! Just start using Black and White Bleaching Cream as directed and see your skin get a lighter, brighter, softer look. Its bleaching action works effectively inside your skin. Modern science knows no faster way of lightening skin. Be sure to start using Black and White Bleaching Cream this very day!"'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"New faster-working formula"; "Discover Ultra Nadinola for a brighter, lighter, more even-toned look in all these beauty areas"; "new Ultra Nadinola for skin discolorations"',
      '"Helps keep skin soft and smooth…a powder base…not greasy…vanishes immediately…for skin blemishes…"; "Contains the new wonderful ingredient Persu - Persu is Drake Laboratories, Inc. trademark for stabilized hydroquinone compounds."'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"Lighter, clearer skin too! May be yours in 7 days plus a softer, lovelier complexion!"',
      '"Wonderful new ULTRA Nadinola is a specially forumated moisturizing cream that brings you the lightening, brightening magic of Hydroquinone. It actually seeks out and fades dark areas to a softer, more golden-toned glow. Meanwhile, it smooths and freshens the clearer areas. Remarkable improvement usually comes after using a single tube of ULTRA Nadinola. After that, occasional use may be needed to keep your lighter, brighter, uniform skin tone. ULTRA Nadinola also contains a \'sun screen\' to help protect sun-sensitive skin and preserve its new freshness and clearness. Start your new complexion today. Extensive clinical testing under doctors\' supervision has proved ULTRA Nadinola safe and effective for normal skin."; "YOUR FACE: ULTRA Nadinola does what most cosmetics can\'t even pretend to do. It brings you lighter, brighter skin beauty - fades dark areas, weathered spots and other such discolorations.  At the same time its special moisturizing formula adds precious smoothing moisture to dry skin."; "YOUR HANDS: Nothing more deserves beauty care than your hands. And nothing makes you look older than dark spots on them. Take advantage of ULTRA Nadinola\'s lightening and clearing action. It even works on deep-seated \'age spots\' and fades them to more even-toned, youthful looking beauty."; "YOUR ELBOWS AND KNEES: Don\'t think people don\'t see your elbows. And you know they look at your knees. If neglected, these \'friction areas\' often have a rough, dark, smudgy appearance. ULTRA Nadinola helps smooth and soften them - lightens darker pigment areas to a brigher, lovelier tone."; "YOUR SHOULDERS AND NECK: This is where the first signs of age often make their appearance. Shoulders become blotchy and freckled. Rusty discolorations creep up the side of the neck. ULTRA Nadinola works directly on these darkened areas to produce a brighter, more even-toned glowing effect."'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"Lighter, clearer skin too! May be yours in 7 days plus a softer, lovelier complexion!"',
      '"Dr. Fred Palmer\'s Skin Whitener, an exclusive formula for a lighter and smoother look...Your skin will seem to glow as it suddenly comes alive! Easy and pleasant to use…it contains effective ingredients repeatedly prescribed by Doctors for skin care. Also a highly recommended aid for removing blackheads and refining enlarged pores. This treatment is attested to and proven effective by thousands of women the world over. Try it...your skin will be radiant and you will be too!"'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"Lighter, clearer skin too! May be yours in 7 days plus a softer, lovelier complexion!"',
      '"And you, too can have a glamorous complexion! Just start using Black and White Bleaching Cream as directed and see your skin get a lighter, brighter, softer look. Its bleaching action works effectively inside your skin. Modern science knows no faster way of lightening skin. Be sure to start using Black and White Bleaching Cream this very day!"'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"Lighter, clearer skin too! May be yours in 7 days plus a softer, lovelier complexion!"',
      '"Helps keep skin soft and smooth…a powder base…not greasy…vanishes immediately…for skin blemishes…"; "Contains the new wonderful ingredient Persu - Persu is Drake Laboratories, Inc. trademark for stabilized hydroquinone compounds."'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"Lighter, brighter skin is irresistable"',
      '"Wonderful new ULTRA Nadinola is a specially forumated moisturizing cream that brings you the lightening, brightening magic of Hydroquinone. It actually seeks out and fades dark areas to a softer, more golden-toned glow. Meanwhile, it smooths and freshens the clearer areas. Remarkable improvement usually comes after using a single tube of ULTRA Nadinola. After that, occasional use may be needed to keep your lighter, brighter, uniform skin tone. ULTRA Nadinola also contains a \'sun screen\' to help protect sun-sensitive skin and preserve its new freshness and clearness. Start your new complexion today. Extensive clinical testing under doctors\' supervision has proved ULTRA Nadinola safe and effective for normal skin."; "YOUR FACE: ULTRA Nadinola does what most cosmetics can\'t even pretend to do. It brings you lighter, brighter skin beauty - fades dark areas, weathered spots and other such discolorations.  At the same time its special moisturizing formula adds precious smoothing moisture to dry skin."; "YOUR HANDS: Nothing more deserves beauty care than your hands. And nothing makes you look older than dark spots on them. Take advantage of ULTRA Nadinola\'s lightening and clearing action. It even works on deep-seated \'age spots\' and fades them to more even-toned, youthful looking beauty."; "YOUR ELBOWS AND KNEES: Don\'t think people don\'t see your elbows. And you know they look at your knees. If neglected, these \'friction areas\' often have a rough, dark, smudgy appearance. ULTRA Nadinola helps smooth and soften them - lightens darker pigment areas to a brigher, lovelier tone."; "YOUR SHOULDERS AND NECK: This is where the first signs of age often make their appearance. Shoulders become blotchy and freckled. Rusty discolorations creep up the side of the neck. ULTRA Nadinola works directly on these darkened areas to produce a brighter, more even-toned glowing effect."'],
     ["Lighter, lovelier skin today…the American way!'",
      "…to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and",
      '"Lighter, brighter skin is irresistable"',
      '"Dr. Fred Palmer\'s Skin Whitener, an exclusive formula for a lighter and smoother look...Your skin will seem to glow as it suddenly comes alive! Easy and pleasant to use…it contains effective ingredients repeatedly prescribed by Doctors for skin care. Also a highly recommended aid for removing blackheads and refining enlarged pores. This treatment is attested to and proven effective by thousands of women the world over. Try it...your skin will be radiant and you will be too!"']]




```python
both_claims_phrase = []
for c1, p1, c2, p2 in both_claims_phrase_list:
    cp = str(c1) + str(p1) + str(c2) + str(p2)
    both_claims_phrase.append(cp)
```


```python
cleaned_doc = [preprocess(doc).split() for doc in both_claims_phrase]
```


```python
dictionary = corpora.Dictionary(cleaned_doc)
```


```python
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleaned_doc]
```


```python
lda = gensim.models.ldamodel.LdaModel
```


```python
ldamodel = lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=20)
```


```python
both_topics = ldamodel.print_topics(num_topics=10, num_words=5)
print(both_topics)
```

    [(0, '0.082*"year" + 0.074*"skin" + 0.042*"large" + 0.034*"look" + 0.030*"cream"'), (1, '0.088*"skin" + 0.054*"thats" + 0.046*"nadinola" + 0.042*"beautiful" + 0.038*"oily"'), (2, '0.072*"skin" + 0.031*"treatment" + 0.028*"peelerpak" + 0.025*"6" + 0.024*"day"'), (3, '0.089*"skin" + 0.033*"dark" + 0.032*"help" + 0.027*"tone" + 0.020*"work"'), (4, '0.068*"skin" + 0.049*"palmer" + 0.033*"formula" + 0.031*"aid" + 0.030*"using"'), (5, '0.046*"ultra" + 0.039*"skin" + 0.038*"spot" + 0.036*"dark" + 0.035*"lemon"'), (6, '0.093*"glow" + 0.083*"bleach" + 0.061*"skin" + 0.041*"cream" + 0.034*"beauty"'), (7, '0.087*"ambi" + 0.078*"skin" + 0.037*"cream" + 0.035*"tone" + 0.032*"artra"'), (8, '0.038*"result" + 0.031*"superficial" + 0.027*"skin" + 0.026*"formulated" + 0.024*"eye"'), (9, '0.115*"esoterica" + 0.054*"skin" + 0.036*"surface" + 0.032*"cream" + 0.030*"hand"')]



```python
both_topic_list=[]
for score,topic in both_topics:
    topic_vals=[tuple(top.split("*")) for top in topic.split(" +")]
    both_topic_list.append(topic_vals)
```


```python
both_topic_list
```




    [[('0.082', '"year"'),
      (' 0.074', '"skin"'),
      (' 0.042', '"large"'),
      (' 0.034', '"look"'),
      (' 0.030', '"cream"')],
     [('0.088', '"skin"'),
      (' 0.054', '"thats"'),
      (' 0.046', '"nadinola"'),
      (' 0.042', '"beautiful"'),
      (' 0.038', '"oily"')],
     [('0.072', '"skin"'),
      (' 0.031', '"treatment"'),
      (' 0.028', '"peelerpak"'),
      (' 0.025', '"6"'),
      (' 0.024', '"day"')],
     [('0.089', '"skin"'),
      (' 0.033', '"dark"'),
      (' 0.032', '"help"'),
      (' 0.027', '"tone"'),
      (' 0.020', '"work"')],
     [('0.068', '"skin"'),
      (' 0.049', '"palmer"'),
      (' 0.033', '"formula"'),
      (' 0.031', '"aid"'),
      (' 0.030', '"using"')],
     [('0.046', '"ultra"'),
      (' 0.039', '"skin"'),
      (' 0.038', '"spot"'),
      (' 0.036', '"dark"'),
      (' 0.035', '"lemon"')],
     [('0.093', '"glow"'),
      (' 0.083', '"bleach"'),
      (' 0.061', '"skin"'),
      (' 0.041', '"cream"'),
      (' 0.034', '"beauty"')],
     [('0.087', '"ambi"'),
      (' 0.078', '"skin"'),
      (' 0.037', '"cream"'),
      (' 0.035', '"tone"'),
      (' 0.032', '"artra"')],
     [('0.038', '"result"'),
      (' 0.031', '"superficial"'),
      (' 0.027', '"skin"'),
      (' 0.026', '"formulated"'),
      (' 0.024', '"eye"')],
     [('0.115', '"esoterica"'),
      (' 0.054', '"skin"'),
      (' 0.036', '"surface"'),
      (' 0.032', '"cream"'),
      (' 0.030', '"hand"')]]




```python
G = nx.Graph()
top_num = 0
for topics in both_topic_list:
    top_num += 1
    G.add_node("Topic "+str(top_num))
    for topic in topics:
        G.add_node(topic[1])
        G.add_edge("Topic "+str(top_num), topic[1])
plt.figure(figsize=(18,9))
plt.suptitle("LDA Topic Model of Drum and Ebony Magazine skin bleaching ads", fontsize=16)
nx.draw(G, with_labels=True)
plt.savefig("./drum_ebony_LDA_network.png")
```


    
![png](output_156_0.png)
    


## Locating Specific Ads

Drum: 
1. "laboratory proven for effectiveness and safe use on normal skin so soothing and refreshing too artras"
2. "'ugly', 'spots', 'and', 'pimples'"
3. "beautiful bride forthat want a lighter sa beautiful bride for people that want a lighter"
4. "give you the smooth clear lighter skin youve notices successful people have hilite lightens and smooths skin"
5. "whitens', 'your', 'skin', 'now'"
6. "more every day american scientist made artralightens and good things happen to a pretty girl brightens skin lightens fr"
7. "smoother protects your skin against harsh sunlight successful people use ambi worried by little pimples and spotskar"
8. "by little pimples and spotskarroo takes these away too successful people use ambi keep my complexion looking light clear"
9. "n give you the smooth clear lighter skin youve notices successful people have karroo morning karoo night makes you"
10. "of your skin you too can look as lovely as a beautiful bride you will be more attractive more desirable will admire"


```python
drum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Product name</th>
      <th>chemical/active ingredient</th>
      <th>Claims</th>
      <th>Legal issues and Politics</th>
      <th>Race</th>
      <th>Age</th>
      <th>Advertising strategy *quotes-catch phrase*</th>
      <th>Size of Advert</th>
      <th>Pg. reference ( marked)</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1965-01-01</th>
      <td>ARTRA skin tone cream</td>
      <td>Hydroquinone</td>
      <td>…to make their skin lighter and lovelier…lovel...</td>
      <td>Black model and white pharmacist/doctor</td>
      <td>Black</td>
      <td>20+</td>
      <td>Lighter, lovelier skin today…the American way!'</td>
      <td>full pg.</td>
      <td>pg. 2</td>
      <td>The ad says that the cream was developed after...</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>brightens skin.' '…lightens from the first day...</td>
      <td>n/a</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cream your skin lighter and brighter with ama...</td>
      <td>NaN</td>
      <td>pg. 2</td>
      <td>Ad states that it is a medicated beauty bar. S...</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>immediately.' '…keeps skin beautiful and clean...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>Artra beauty bar</td>
      <td>Hydroquinone</td>
      <td>…mild and gentle…keeps skin free from blemishe...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20+</td>
      <td>Medicated soap for complexion care</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1965-01-01</th>
      <td>Aloma Crème blanche</td>
      <td>unnamed</td>
      <td>…clears and lightens the skin, smooth's away b...</td>
      <td>Black model</td>
      <td>Black</td>
      <td>20-35</td>
      <td>good things happen to a pretty girl</td>
      <td>full pg.</td>
      <td>NaN</td>
      <td>Ad states that using this product will increas...</td>
    </tr>
  </tbody>
</table>
</div>




```python
catch_phrase_drum.head()
```




    Year
    1965-01-01      Lighter, lovelier skin today…the American way!'
    1965-01-01     Cream your skin lighter and brighter with ama...
    1965-01-01                   Medicated soap for complexion care
    1965-01-01                  good things happen to a pretty girl
    1965-01-01                                            See notes
    Name: Advertising strategy *quotes-catch phrase*, dtype: object




```python
claims_drum.head()
```




    Year
    1965-01-01    …to make their skin lighter and lovelier…lovel...
    1965-01-01    brightens skin.' '…lightens from the first day...
    1965-01-01    immediately.' '…keeps skin beautiful and clean...
    1965-01-01    …mild and gentle…keeps skin free from blemishe...
    1965-01-01    …clears and lightens the skin, smooth's away b...
    Name: Claims, dtype: object




```python
for claim in range(len(drum['Claims'].dropna().str.contains('success'))):
    if type(drum['Claims'][claim]) == float:
        continue
    if 'success' in drum['Claims'][claim]:
        print(drum.index[claim],drum['Claims'][claim])
    
```

    1965-07-19 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1966-08-01 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1966-09-19 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1966-10-01 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1966-11-01 00:00:00 …lovey, light complexion' '…success comes from regular Karroo cream care
    1967-01-01 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1967-03-01 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1967-05-01 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1967-07-01 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1967-09-01 00:00:00 …lighter, clearer complexion with Ambi…' 'Ambi can give you the smooth, clear, lighter skin you've notices successful people have.'
    1968-06-21 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1968-08-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1968-09-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1968-10-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1968-11-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1968-11-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-01-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-02-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-03-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-04-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-05-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-06-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-07-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-08-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-09-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-11-21 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1969-12-21 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1970-01-01 00:00:00 ...lighter, clearer complexion'  '…smooth, clear, lighter skin you've noticed successful people have.'
    1970-08-01 00:00:00 …lovey, light complexion' '…success comes from regular Karroo cream care
    1970-09-01 00:00:00 …lovey, light complexion' '…success comes from regular Karroo cream care
    1970-10-01 00:00:00 …lovey, light complexion' '…success comes from regular Karroo cream care
    1970-12-01 00:00:00 …lovey, light complexion' '…success comes from regular Karroo cream care
    1971-04-01 00:00:00 …lovey, light complexion' '…success comes from regular Karroo cream care



```python
drum.loc['1971-08-01']['Claims']
```




    Year
    1971-08-01                      stop ugly spots and blemishes' 
    1971-08-01                                                  NaN
    1971-08-01    the most powerful name in skin lightening crea...
    1971-08-01    acting skin lightening cream' 'Hollywood Seven...
    1971-08-01    lightens, smooth's and clears skin' "super-sun...
    1971-08-01                                  fast and effective'
    1971-08-01    …improved ARTRA skin cream…with latest modern ...
    1971-08-01    ;works safely to make you lighter, brighter an...
    1971-08-01                                                  NaN
    1971-08-01    contains fast acting Ambi ingredients to light...
    1971-08-01    burning rays of the sun' 'makes you lighter an...
    1971-08-01    lightens, smooth's and clears skin ' "super-su...
    1971-08-01                                                  NaN
    1971-08-01    lightens skin overnight safely' 'during the ni...
    1971-08-01    …safe ingredients goes deep into skin & cleans...
    1971-08-01                             of pimples and blemishes
    1971-08-01    lovely beauty queen complexion' 'makes skin sm...
    1971-08-01    …Karroo is best" "…keeps face looking lovely t...
    Name: Claims, dtype: object




```python
try:
    for i in range(len(drum.Claims)):
        if 'skin' in drum.Claims[i]:
            print(drum.Claims[i])
            print(drum.index[i])
except TypeError:
    pass
```

    …to make their skin lighter and lovelier…lovelier and lighter…a little more every day' '…American scientist made artra…lightens and
    1965-01-01 00:00:00
    brightens skin.' '…lightens from the first day.' '…vanishes into skin instantly…starts working, starts lightening and brightening your skin
    1965-01-01 00:00:00
    immediately.' '…keeps skin beautiful and clean, makes it smooth and lovely
    1965-01-01 00:00:00
    …mild and gentle…keeps skin free from blemishes and pimples
    1965-01-01 00:00:00
    …clears and lightens the skin, smooth's away blemishes and spots, softens the skin'
    1965-01-01 00:00:00



```python
# Searching in whole column for term

try:
    for i in range(len(drum.Claims)):
        if drum.Claims is str:
            if "bride" in drum.Claims[i]:

            # indx will store the tuple having that 
            # particular value in column.
                indx = i
        # below line will print that tuple
        drum.iloc[indx]
except NameError:
    print("Not found.")
```

    Not found.



```python
ebony.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Region</th>
      <th>Source</th>
      <th>Product name</th>
      <th>chemical/active ingredient</th>
      <th>Structure</th>
      <th>Chemistry</th>
      <th>Claims</th>
      <th>Legal issues and Politics</th>
      <th>Race</th>
      <th>Age</th>
      <th>Advertising strategy *quotes-catch phrase*</th>
      <th>Size of Advert</th>
      <th>Pg reference (quentin marked)</th>
      <th>Notes</th>
    </tr>
    <tr>
      <th>Year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>Ebony</td>
      <td>Long Aid Bleach and Glow</td>
      <td>unnamed</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>" - wakes up dark, dull complexion! Conceals u...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1/2 pg</td>
      <td>63</td>
      <td>small part of 1.2 pg ad for Long Aid hair prod...</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mercolized Wax Cream</td>
      <td>ammoniated mercury; zinc oxide</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"If your skin doesn't look actually lighter af...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Mercolized Wax Cream guarantees lighter looki...</td>
      <td>1/4 pg</td>
      <td>72</td>
      <td>ingredient on image of product; not mentioned ...</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Black and White Bleaching Cream</td>
      <td>unnamed</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"And you, too, can have a glamorous complexion...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Lighter, brighter skin is irresistable"</td>
      <td>1/8 pg</td>
      <td>83</td>
      <td>drawing of white man and white woman in ad</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Nadinola Bleaching Cream</td>
      <td>"wonder-working A-M"</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Don't let dull, dark skin rob you of romance....</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"LIFE IS MORE FUN when your complexion is clea...</td>
      <td>full pg</td>
      <td>91</td>
      <td>two types advertised - oily and dry skin</td>
    </tr>
    <tr>
      <th>1960-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dr. Fred Palmer's Double Strength Skin Whitener</td>
      <td>zinc phenolsulfonate</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"Yes in just 7 days be delighted how fast and ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>"DR. FRED PALMER'S IN JUST 7 DAYS MUST GIVE YO...</td>
      <td>1/8 pg</td>
      <td>108</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


