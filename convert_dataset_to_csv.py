import regex as re
import pandas as pd

def parse_review_file(file_name):
    """ This function parses a .review file to separate the different reviews, before parsing each review separately."""
    data_str = ""
    with open(file_name, "r") as file:
        for line in file.readlines():
            data_str += line
    return data_str

def parse_review(review_to_parse):
    """ This function is used on each review to parse them."""
    tags_list = ["unique_id", "asin", "product_name", "product_type", "helpful", "rating", "title", "date", "reviewer", "reviewer_location", "review_text"]
    review_data = {}
    for tag in tags_list:
        pattern = f"<{tag}.*>\\n((.|\\n)*?)<\\/{tag}>\\n"
        review_data[tag] = re.sub("\n", "", re.findall(pattern, review_to_parse)[0][0])
    return review_data

if __name__=="__main__":
    product_type_list = ["books", "dvd", "electronics", "kitchen_&_housewares"]
    sentiment_list = ["negative", "positive"]
    for product_type in product_type_list:
        for sentiment in sentiment_list:
            file_name = f"sorted_data_acl/{product_type}/{sentiment}.review"
            text = parse_review_file(file_name)
            pattern_review = "<review.*>\n((.|\n)*?)<\/review>\n"
            matches = re.findall(pattern_review, text)
            matches = [_[0] for _ in matches]
            parsed_reviews = [parse_review(_) for _ in matches]
            
            review_text_list = []
            rating_list = []

            for review in parsed_reviews:
                review_text_list.append(review["review_text"])
                rating_list.append(review["rating"])

            dict_for_df = {
                "rating" : rating_list,
                "review_text" : review_text_list,
            }

            df = pd.DataFrame(dict_for_df)
            df.to_csv(f"{file_name[:-7]}.csv", index=False)