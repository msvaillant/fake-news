#!/usr/bin/env python
# coding: utf-8


import os
import re
from Color import *
import sys
import praw
import requests
from bs4 import BeautifulSoup



class RedditScrapper():
    def __init__(self,subredditlist, c_id,secret,nbnews = 500):
        self.subreddits = subredditlist
        self.client_id = c_id
        self.client_secret = secret
        self.nb_newsOK = 0
        self.nb_news = nbnews
    def get_reddit(self):
        return praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            grant_type='client_credentials',
            user_agent='mytestscript/1.0')


    def get_top(self,subreddit_name):
        print(subreddit_name)
        dirname = 'news'
        os.makedirs(dirname, exist_ok=True)

        # Get top 50 submissions from reddit
        reddit = self.get_reddit()
        top_subs = reddit.subreddit(subreddit_name).top(limit=1000)
        return top_subs

    def create_news(self,subreddit_name):

        top_subs=self.get_top(subreddit_name)
        # Remove those submissions that belongs to reddit
        subs = [sub for sub in top_subs if not sub.domain.startswith('self.')]

        count = 1000
        while subs and count > 0 and self.nb_newsOK < (self.nb_news/len(self.subreddits)):
            sub = subs.pop(0)
            article = self.get_article(sub.url)
            if article:
                text = '\n\n'.join(article['content'])
                filename = str(self.nb_newsOK) + '.news'


                if len(text)!=0:
                    self.nb_newsOK +=1

                    try:
                        open(os.path.join(dirname, filename), 'w').write(text)
                        display(filename+"\n",'yellow')
                    except UnicodeEncodeError:
                        pass
                    count -= 1


    def get_article(self,url):
        display('  - Retrieving %s' % url,'blue')
        try:
            res = requests.get(url)
            if (res.status_code == 200 and 'content-type' in res.headers and
                    res.headers.get('content-type').startswith('text/html')):
                article = self.parse_article(res.text)
                print('      => done, title = "%s"' % str(self.nb_newsOK))
                return article
            else:

                display('      x fail or not html',"red")

        except Exception as e:
            print(e)
            pass


    def parse_article(self,text):
        soup = BeautifulSoup(text, 'html.parser')

        # find the article title
        h1 = soup.body.find('h1')

        # find the common parent for <h1> and all <p>s.
        root = h1
        while root.name != 'body' and len(root.find_all('p')) < 5:
            root = root.parent

        if len(root.find_all('p')) < 5:
            return None

        # find all the content elements.
        ps = root.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre'])
        ps.insert(0, h1)
        content = [self.tag2md(p) for p in ps]

        return {'title': h1.text, 'content': content}


    def tag2md(self,tag):
        if tag.name == 'p':
            return tag.text
        elif tag.name == 'h1':
            return f'{tag.text}\n{"=" * len(tag.text)}'
        elif tag.name == 'h2':
            return f'{tag.text}\n{"-" * len(tag.text)}'
        elif tag.name in ['h3', 'h4', 'h5', 'h6']:
            return f'{"#" * int(tag.name[1:])} {tag.text}'
        elif tag.name == 'pre':
            return f'```\n{tag.text}\n```'


    def main(self):

        for sr in self.subreddits:
            print('Scraping /r/%s...' % sr)
            self.create_news(sr)
        print("DONE\n")



# In[3]:

fconf = open("Config.conf","r")
raw = fconf.read()
id= raw.split(':')[0]
secret=raw.split(':')[1]
scrapper = RedditScrapper(['The_Donald'],id,secret,int(sys.argv[1]))
scrapper.main()
