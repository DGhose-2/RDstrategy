def name_cleaner(df):
    from cleanco import cleanco
    
    df_names = df['name'].fillna(' ')
    
    # removal of text between parentheses
    df_names = df_names.str.replace(r"\(.*\)","")
    
    # 'AND' and '&' are equivalent
    df_names = df_names.str.replace(' AND ', ' & ')
    
    # cleaning utilities from cleanco package (takes off suffixes from a database)
    df_names = df_names.str.replace('.', '')
    df_names = df_names.apply(lambda x: cleanco(x).clean_name() if type(x)==str else x)
    
    # make all names lower-case
    df_names = df_names.str.lower()
    
    return df_names

def string_matcher(col1, col2, sim_thresh=0.95):
    from string_grouper import match_strings, match_most_similar, group_similar_strings, StringGrouper
    matches = match_strings(col1, col2, min_similarity = sim_thresh)
    return matches

# removing of same-named companies
def company_clean(df):
    df = df.sort_values('rank').drop_duplicates(subset='Cleannames', keep='first')
    return df

def info_merger(df1, df2,
                match_table,
                df1_desc,
                df2_info,
                df_links):
    # merge matches with db1 and specify the columns you want to use as key
    df_matches = match_table.reset_index().merge(df1, left_on=['left_side'], right_on=['Cleannames'], how="left").set_index("index")
    
    # merge matches (already matched with db1) again with db2 and specify the new columns you want to use as key
    df_matches = df_matches.reset_index().merge(df2, left_on=['right_side'], right_on=['Cleannames'], how="left", suffixes=["_TDP", "_UKRI"]).set_index("index")
    
    #include TDP organization descriptions:
    df_matches = df_matches.reset_index().merge(df1_desc, left_on=['uuid'], right_on=['uuid'], how="left").set_index("index")
    
    #select output columns
    df_matches_red = df_matches[['left_side', 'Cleannames_TDP', 'right_side', 'Cleannames_UKRI', 'uuid', 'gtrorguuid', 'rank_x', 'category_groups_list', 'category_list', 'short_description', 'description']]
    
    
    # Now we build the orgs-project links dataframe:
    
    # get in the organization description data
    orgProjectLinks_matches = df_links.reset_index().merge(df_matches_red, left_on=['orguuid'], right_on=['gtrorguuid'], how="left").set_index("index")

    # remove rows with any NA, to streamline preprocessing (still leaving a large number of organization-project links)
    orgProjectLinks_matches = orgProjectLinks_matches.dropna()
    
    # include project information
    orgProjectLinks_matches = orgProjectLinks_matches.reset_index().merge(df2_info, left_on=['projectuuid'], right_on=['projectuuid'], how="left", suffixes=["_org","_project"]).set_index("index").drop_duplicates(subset=["orguuid", "projectuuid"])
    
    # compute total project count
    orgProjectLinks_matches['count'] = orgProjectLinks_matches.groupby('orguuid')['orguuid'].transform('count')
    
    # sort by count, for easy viewing
    orgProjectTexts = orgProjectLinks_matches.sort_values(['count', 'orguuid', 'startdate'], ascending=[False, True, True])
    
    return orgProjectTexts
