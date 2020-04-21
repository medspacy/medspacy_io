import spacy

from medspacy_io.consumer.doc_consumer import DocConsumer, print_rows

nlp = spacy.load("en_core_web_sm")



d_consumer = DocConsumer("doc")
ent_consumer = DocConsumer("ent")
token_consumer = DocConsumer("token")

# first few paragraphs from ny times article clicked from front page
# https://www.nytimes.com/2020/04/18/us/texas-protests-stay-at-home.html
doc = nlp("In a public act of defiance, several dozen protesters in Texas converged on the steps of the Capitol building in Austin on Saturday to call for the reopening of the state and the country during the coronavirus pandemic. The 'You Can’t Close America' rally rode a wave of similar protests at statehouses and in city streets this past week, with people also gathering on Saturday in Indianapolis; Carson City, Nev.; Annapolis, Md.; and Brookfield, Wis. With more than 22 million unemployment claims nationwide in the past four weeks, some conservatives have begun voicing displeasure with the moribund economy. Many businesses have been shuttered in an attempt to slow the spread of the virus, which has killed more than 34,000 people in the United States. The protests have been encouraged by President Trump, but polls show that most Americans support restrictions meant to combat the virus. This month, a poll by Quinnipiac University found that 81 percent of registered voters supported a theoretical nationwide stay-at-home order, including 68 percent of Republicans polled. Others, though, are openly violating the stay-at-home orders replicated by governors across the country by assembling to express their dissatisfaction. In Austin, at least 100 people gathered on the statehouse grounds in hats and shirts with President Trump’s slogan, “Make America Great Again.” Some carried American flags, and few wore masks that are mandated by the city. There were cheers at the sight of Alex Jones, the founder of the website Infowars, which traffics in conspiracy theories. There were chants of 'Fire Fauci,' in reference to Dr. Anthony S. Fauci, the federal government’s top infectious disease expert.")

print("CONSUMING DOCUMENT...")
print(list(d_consumer.attrs))
print_rows(d_consumer.get_rows(doc))

print("\nCONSUMING ENTITIES...")
print(list(ent_consumer.attrs))
print_rows(ent_consumer.get_rows(doc))

print("\nCONSUMING TOKENS...")
print(list(token_consumer.attrs))
print_rows(token_consumer.get_rows(doc))