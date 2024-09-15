import sys
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.DEBUG)

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """ Load the messages and categories data and merge them into a single dataframe.

    Args:
        messages_filepath (str): Path to the messages file
        categories_filepath (str): Path to the categories file

    Returns:
        pd.DataFrame: Merged dataframe
    """
    # Load the data
    messages_df = pd.read_csv(messages_filepath)
    logging.info("Messages data loaded")
    categories_df = pd.read_csv(categories_filepath)
    logging.info("Categories data loaded")
    logging.debug(f"Messages shape: {messages_df.shape}")
    logging.debug(f"Categories shape: {categories_df.shape}")


    # Merge the two dataframes into main_df
    main_df = messages_df.merge(categories_df, on='id', how='inner')
    logging.info("Data merged")
    logging.debug(f"Merged data shape: {main_df.shape}")
    
    return main_df

    
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean the data by splitting the ctext by ';' and converting the values to 0 or 1.

    Args:
        df (pd.DataFrame): Dataframe to clean

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Split the categories column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    logging.info("Categories split into separate columns")
    logging.debug(f"Categories shape: {categories.shape}")

    # Get the unique category names by taking the first row and removing the last two characters
    row = categories.iloc[0,:]
    category_colnames = list(row.apply(lambda x: x[:-2]).unique())
    categories.columns = category_colnames
    logging.info("Category column names set")
    logging.debug(f"Category column names: {category_colnames}")

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # replace 2 with 1
    categories.replace(2, 1, inplace=True)
    assert categories.isin([0, 1]).all().all(), "Categories not converted to 0 or 1"
    logging.info("Category values converted to 0 or 1")
    logging.debug(f"Categories converted: {categories.dtypes}")

    # Drop the original categories column from `df`
    df.drop(labels=['categories'], inplace=True, axis=1)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    logging.info("Data concatenated")
    logging.debug(f"Data shape after concatenation: {df.shape}")
    assert df.shape[0] == categories.shape[0], "Dataframes not concatenated correctly"
    
    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """ Save the cleaned data to a SQLite database.

    Args:
        df (pd.DataFrame): Cleaned dataframe
        database_filename (str): Name of the database file
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('main', engine, index=False, if_exists='replace')
    logging.info("Data saved to database")
    logging.debug(f"Data saved to database: {database_filename}")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()