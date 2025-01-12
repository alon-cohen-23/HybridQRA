#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:17:42 2024

@author: aloncohen
"""

from OpenAI_api_conn import Azure_OpenAI_api, OpenAI_api
import pandas as pd
import yaml

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

openai_config = config['openai']    

def critic_llm_validation (row) -> str:
    """
    the function ask the llm to correct the RAG answer if needed under specific rules,
    in order to get the most accurate results the prompt contains
    the instructions (rules), and 3 few shots that represent 3 different situations.
     
    Parameters
    ----------
    row : row from the synthetic ragas_df generated at generate_testset.llama_index_ragas_df.

    Returns
    -------
    critic_answer : the llm output after it got the critic prompt.

    """
    
    instructions = """If it's possible, fix the answer of the given question base on the context given.
     generate your answer based of few rules:
        1. Very important! if the given answer is already correct don't change it, just return the same answer! Keep in mind that most of the times you would not need to change anything!!!
        2. Very important! if you change anything, generate your answer only based on the given context!
        3. Keep your answer short and straightforward without any unnecessary additions.
        4. if there is no valid answer to the question based on the given context just return 'There is no answer
        """
        
    few_shots = [
            
            """When the given answer is valid:
            question: How many home runs did Mays hit in one game in Milwaukee in 1961?
            contexts: ['[{\'document\': \'One night in Milwaukee in 1961, Mays hit four home runs in one game.\', \'metadata\': {\'content_publish_date\': \'Jun 18, 2024, 09:55 PM ET\',\'title\': \'Willie Mays at 90 -- He was Steph Curry, Michael Jordan, Simone Biles and Mikhail Baryshnikov\'}}, {\'document\': \'In his 22-year career, Mays led the NL in home runs four times, and when he retired, his 660 home runs ranked third in big league history; he now ranks sixth behind Bonds, Hank Aaron, Ruth, Alex Rodriguez and Albert Pujols. He also finished his career with 3,283 hits (12th all time) and 1,903 RBIs (12th all time).\', \'metadata\': {\'content_publish_date\': \'Jun 18, 2024, 09:03 PM ET\',\'title\': "Legendary outfielder Willie Mays, \'Say Hey Kid,\' dies at 93"}}, {\'document\': \'But the layoff from professional baseball did not affect him. Mays won the first of his two career NL MVP awards that season, leading the league in batting at .345 and hitting 41 home runs to go along with 110 RBIs. Mays won his other NL MVP in 1965.\', \'metadata\': {,\'content_publish_date\': \'Jun 18, 2024, 09:03 PM ET\',\'title\': "Legendary outfielder Willie Mays, \'Say Hey Kid,\' dies at 93"}}]']
            answer: Mays hit four home runs in one game in Milwaukee in 1961.
            expected output: Mays hit four home runs in one game in Milwaukee in 1961.""",
    
            """when there is not a good answer to the question in the context:
                question: Why were there no substantial contract offers for Kyrie Irving last offseason?
                contexts: ['[{\'document\': \'Remember last offseason when Kyrie Irving entered free agency and there were no teams willing to offer a substantial contract other than the Dallas Mavericks?\', \'metadata\': {\'content_publish_date\': \'Jun 17, 2024, 07:15 AM ET\', \'title\': \'2024 NBA free agency: Players to sign, contract predictions\'}}, {\'document\': "Irving signed a three-year, $120 million deal, and the Mavericks were ridiculed because they were considered to be bidding against themselves. Irving\'s contract illustrated the Mavericks\' willingness to compromise and not have the All-Star feel unwanted, despite their position of leverage.", \'metadata\': {\'author\': \'Bobby Marks\', \'content_publish_date\': \'Jun 17, 2024, 07:15 AM ET\', \'country\': \'us\', \'site\': \'espn\', \'title\': \'2024 NBA free agency: Players to sign, contract predictions\'}}, {\'document\': "Irving\'s 48.2% shooting mark from the field coming into Game 5 against Minnesota was the best of his playoff career, and his 42.3% from 3 wasn\'t far off his best postseason showing with the Cavs. Remarkably, if the Finals go at least five games, Irving will have played as many career postseason games with the Mavericks (22) as he did in his time with the Celtics and Brooklyn Nets combined -- evidence of how well things have clicked for him with the Mavs.", \'metadata\': {/'content_publish_date\': \'Jun 3, 2024, 05:39 PM ET\',\'title\': \'2024 NBA Finals: Big questions ahead of Mavericks-Celtics\'}}]']
                answer: There were no substantial contract offers for Kyrie Irving last offseason because no teams other than the Dallas Mavericks were willing to make such offers.
                expected output: There is no answer.""",
                
            """when the answer need to be changed:
                question: Why did the Cardinals focus on adding young talent instead of making veteran upgrades during the 2024 offseason?
                contexts: ['[{\'document\': \'The Cardinals are still waiting to make their move. The 2024 offseason was about adding young talent -- the Cardinals made 12 draft picks, including seven in the first three rounds -- instead of making veteran upgrades to a generally lacking roster.\', \'metadata\': {\'author\': \'Seth Walder\', \'content_publish_date\': \'Jun 18, 2024, 07:00 AM ET\', \'country\': \'us\', \'site\': \'espn\', \'title\': \'2024 NFL offseason: Grading moves, changes for all 32 teams\'}}, {\'document\': "But what I liked most about the Chargers\' offseason was their restraint. They entered the offseason with an aging, expensive roster and instead of trying to gain some short-term wins by borrowing resources from the future, the Chargers largely set up the roster to use this season as a steppingstone for 2025 and beyond. They made minor moves in free agency -- signing veteran players such as linebacker Denzel Perryman, running back Gus Edwards, center Bradley Bozeman and tight end Hayden Hurst -- released high-priced receiver Mike Williams and traded receiver Keenan Allen, another costly veteran. They retained their two veteran edge rushers, Joey Bosa and Khalil Mack (a bit of a surprise), but got them to take pay cuts.", \'metadata\': {\'author\': \'Seth Walder\', \'content_publish_date\': \'Jun 18, 2024, 07:00 AM ET\', \'country\': \'us\', \'site\': \'espn\', \'title\': \'2024 NFL offseason: Grading moves, changes for all 32 teams\'}}, {\'document\': \'It\\\'s hurting Dallas again in 2024. The uncertainty and massive amount of money due to Prescott, Lamb and Parsons on their new deals have caused the Cowboys to essentially sit out free agency this offseason. They let offensive linemen Tyron Smith and Tyler Biadasz, edge rusher Dorance Armstrong and running back Tony Pollard sign elsewhere. They didn\\\'t re-sign Gilmore, and they cut Michael Gallup without adding a veteran replacement. Their only two additions in free agency -- in a year in which Jones says the team is "all-in" -- have been Elliott and linebacker Eric Kendricks, veterans on the backside of their careers. Dallas\\\' roster is worse in 2024 than it was in 2023, and it will likely be worse in 2025 than it was in 2024, with or without Prescott on the roster.\', \'metadata\': {\'author\': \'Bill Barnwell\', \'content_publish_date\': \'Jun 19, 2024, 06:15 AM ET\', \'country\': \'us\', \'site\': \'espn\', \'title\': "Dak Prescott\'s Cowboys future: New contract or test free agency?"}}]']
                answer: The Cardinals focused on adding young talent instead of making veteran upgrades during the 2024 offseason because they wanted to win the championship. 
                expected output: The Cardinals focused on adding young talent instead of making veteran upgrades during the 2024 offseason because their roster was generally lacking and they made 12 draft picks, including seven in the first three rounds."""
                ]
    # clean the few_shots from the backslashes      
    few_shots = [shot.replace("\\", "") for shot in few_shots]
    
    question = "question: " + row['question']
    contexts = "contexts: " + row['contexts']
    answer = "answer: "+ row['answer']
    
    llm_input = question +'\n' + contexts +'\n' + answer    
    
    messages = [{"role": "system", "content": instructions},
                {"role": "assistant", "content": "Here are few examples:" +few_shots[0]},     
                {"role": "assistant", "content": few_shots[1]},
                {"role": "assistant", "content": few_shots[2]},
                {"role": "user", "content": "generate an answer based on the instructions you got please : " + llm_input},
                     ]
    
    api_conn = openai_config['conn']
    if api_conn == 'Azure_OpenAI': 
        critic_answer = Azure_OpenAI_api(messages, openai_config['llm'])
    elif api_conn == 'OpenAI':
        critic_answer = OpenAI_api(messages, openai_config['llm'])
    else:
        raise ValueError ("Your api conn must be 'OpenAI' or 'Azure_OpenAI' depend on your key, please change it in the settings or through config.yaml.")
        
    return critic_answer          

def apply_critic_llm_validation (ragas_df) -> pd.DataFrame:
    "apply the critic llm answer from critic_llm_validation to the synthethic ragas df"
    ragas_df['answer'] = ragas_df.apply(critic_llm_validation, axis=1)
    return ragas_df

if __name__ =='__main__':

    # add critic llm answer    
    df = pd.read_csv('/Users/aloncohen/Documents/rag_project/data/testsest/basic_answers.csv')
    critic_llm_validation(df.iloc[0])
    
    print (pd.read_csv('/Users/aloncohen/Documents/rag_project/data/testsest/generated_questions.csv')['contexts'])
    
    df = df[['question','answer','contexts','ground_truth']]
    df = apply_critic_llm_validation(df)
    df.to_csv('50_multi_context_after_critic.csv', index=False)


    # calculate the metrics
    df = pd.read_csv('/Users/aloncohen/Documents/rag_project/evaluation/50_new_multi_context_eval.csv')

    metrics = ['faithfulness',
                'answer_correctness',
                'answer_relevancy',
                'context_entity_recall',
                'context_precision',
                'context_recall',
                'context_relevancy']


    print ('regular_df:')
    average_values = df[metrics].mean()
    print (average_values)

    after_critic_df = pd.read_csv('/Users/aloncohen/Documents/rag_project/evaluation/50_multi_context_after_critic_eval.csv')
    other_rows_df = after_critic_df[~after_critic_df['ground_truth'].str.contains('there is no answer', case=False, na=False)]

    print ('after_critic_df:')
    average_values = other_rows_df[metrics].mean()
    print (average_values)