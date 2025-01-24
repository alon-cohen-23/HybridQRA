#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:08:13 2024

@author: aloncohen
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
import json
import pandas as pd
from typing import List


def collect_espn_urls () -> List:
    """

    Returns
    -------
    A list of all of the urls for the espn articles from espn's main page.

    """
    
    driver = webdriver.Safari()
    url = "https://www.espn.com/"
    driver.get(url)
    
    # Find all <a> elements
    all_a_tags = driver.find_elements(By.TAG_NAME, "a")

    # Extract hrefs and filter out None values
    urls = [element.get_attribute('href') for element in all_a_tags if element.get_attribute('href') is not None]
    
    driver.quit()

    
    urls = list(set(urls))
    # Filter hrefs containing the word "story"
    urls = [element for element in urls if "story" in element]

    return urls

def convert_urls_to_df (urls: List[str]) -> pd.DataFrame:    
    
    """
    Parameters
    ----------
    urls : The list of all of the stories from espn.com, uses the collect_espn_urls function above.

    Returns
    -------
    A pandas df that contains the text of each paragraph from the articles seperately combined with it's metadata. 
    The categories of the metadata are shown in the extracted_metadata dict.

    """
    driver = webdriver.Safari()
    df = pd.DataFrame()
    
    for url in urls:
      
        try:    
            driver.get(url)
          
            # Execute JavaScript to get the __dataLayer object
            dataLayer_script = """
            return JSON.stringify(window.__dataLayer || {});
            """
            
            # Get the JSON string of the __dataLayer object
            dataLayer_json = driver.execute_script(dataLayer_script) 
            dataLayer_dict = json.loads(dataLayer_json)  
                
            # extract the paragraphs text
            paragraphs = driver.find_elements(By.TAG_NAME, "p")
            paragraph_text = [paragraph.text for paragraph in paragraphs]
            
            # Define the structure of extracted_data
            extracted_data = {
                'site': dataLayer_dict.get('site', {}).get('site', 'N/A'),
                'country': dataLayer_dict.get('site', {}).get('country', 'N/A'),
                'title': dataLayer_dict.get('page', {}).get('story_title', 'N/A'),
                'author': dataLayer_dict.get('page', {}).get('author', 'N/A'),
                'content_publish_date': dataLayer_dict.get('page', {}).get('content_publish_date', 'N/A'),
                'league': dataLayer_dict.get('page', {}).get('league', 'N/A'),
                'paragraph_text': paragraph_text  
                        }
            
            df2 = pd.DataFrame([extracted_data])
            df = pd.concat([df, df2], ignore_index=True)
            
        except Exception as e:
            print(f"An error occurred while processing {url}: {str(e)}")            
    
    driver.quit()
    df = df.explode('paragraph_text')
    return df
    
def filter_articles_df (df, min_len = 50) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : The Pandas df given from the convert_urls_to_df function.
    min_len: the min len for the paragraph_text.
    
    Returns
    -------
    A filtered data frame based on the following rules:
        1. paragraph_text contains over the min_len (the defult is 50 charecters).
        2. remove paragraph_text duplicates
        3. remove rows that their paragraph_text contains special chrecters that are used by espn as side notes
        4. remove paragraph_text that contains the word 'cookies' to avoid any cookies notification paragraphs in my df
        5. drop all rows with null values
    """
    df = df.drop_duplicates(subset='paragraph_text')
    
    df = df[df['paragraph_text'].str.len() > min_len]

    df = df[~df['paragraph_text'].str.contains(r'[•|]', regex=True)]
    
    df = df[~df['paragraph_text'].str.contains('cookies')]
    
    df = df.dropna(how='any')
    
    return df

def main ():
    
    lis = collect_espn_urls ()
    df = convert_urls_to_df(lis)
    df = filter_articles_df(df)

    return df

if __name__ =='__main__':
    df = main ()
    df.to_csv('../data/espn/espn_stories.csv', index=False)

   
    
  