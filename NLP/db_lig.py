from lig_nlp import lig_classification
text = "The movie was unhilariously funny, I mean it was bad"
#text = "The movie was visually stunning, but the plot was predictable"
#text = "it is not a mass-market entertainment but an uncompromising attempt by one artist to think about another."
#text = "it 's also heavy-handed and devotes too much time to bigoted views"
#text = "The food tasted awful and the place was dirty"
#text = "It is summer, but the weather is bad"
#text = "The movie was an emotional masterpiece — the storytelling was powerful, the cinematography was breathtaking, and the music added so much depth to every scene"
#text = "i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake"

result = lig_classification(
   text,
    init_path="guided_ig",
    debug=True,
)