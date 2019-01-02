#!/usr/bin/env python
# coding: utf-8

# # Reddit Scrapper
# Inspired by https://hackernoon.com/i-made-a-news-scrapper-with-100-lines-of-python-2e1de1f28f22
# In[1]:

import os
import re
import praw
import requests
from bs4 import BeautifulSoup


# Id : fEHHLFp8KKIUKg
#
# Pass : ngYf56xaw8scuWtS0VaPCWk-cLE
#

# In[2]:


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
        dirname = 'news'
        os.makedirs(dirname, exist_ok=True)

        # Get top 50 submissions from reddit
        reddit = self.get_reddit()
        top_subs = reddit.subreddit(subreddit_name).top(limit=1000)

        # Remove those submissions that belongs to reddit
        subs = [sub for sub in top_subs if not sub.domain.startswith('self.')]

        count = 1000
        while subs and count > 0 and self.nb_newsOK < (self.nb_news/len(self.subreddits)):
            sub = subs.pop(0)
            article = self.get_article(sub.url)
            if article:
                text = '\n\n'.join(article['content'])
                filename = str(self.nb_newsOK) + '.news'
                print(filename + "\n")
                if len(text)!=0:
                    self.nb_newsOK +=1

                    try:
                        open(os.path.join(dirname, filename), 'w').write(text)

                    except UnicodeEncodeError:
                        pass
                    count -= 1


    def get_article(self,url):
        print('  - Retrieving %s' % url)
        try:
            res = requests.get(url)
            if (res.status_code == 200 and 'content-type' in res.headers and
                    res.headers.get('content-type').startswith('text/html')):
                article = self.parse_article(res.text)
                print('      => done, title = "%s"' % str(self.nb_newsOK))
                return article
            else:
                print('      x fail or not html')
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
            self.get_top(sr)
        print("DONE\n")



# In[3]:


scrapper = RedditScrapper(['news','world_news'],'fEHHLFp8KKIUKg','ngYf56xaw8scuWtS0VaPCWk-cLE',60)
scrapper.main()
