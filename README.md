# Pantry Pal

Pantry Pal is a chat agent for young adults living in a city who want to eat well on a budget.
Tell it what you're craving and what groceries you already have, and it finds a recipe, prices
it with Trader Joe's prices (skipping what you already own), suggests cheaper follow-up meals
that reuse your leftovers, and can pair a wine with your dinner.

## Tools

- `search_recipes`: finds recipes by name, ingredient, cuisine, or category (TheMealDB API).
- `estimate_recipe_cost`: prices a recipe ingredient by ingredient from Trader Joe's prices, scaled to your number of servings, and reports the cash to spend and the cost per serving.
- `update_pantry`: records groceries you already own; each one is free for up to 2 more meals by default, then priced normally.
- `suggest_pantry_meals`: finds recipes built around what you already own, cheapest first.
- `suggest_leftover_meals`: finds meals that reuse the leftovers from meals you've chosen and prices only what you'd still need to buy.
- `weekly_summary`: totals what you've spent and the average cost per serving across the meals you've chosen.
- `suggest_wine_pairing`: finds real wines of a style that suits the dish and checks each wine's listed food pairings against it (WineAPI).

## How to use it

Open the deployed app at **https://pantry-pal-git-727379569786.europe-west1.run.app**.

Type in the chat box or click one of the sample prompts on the left. The agent remembers the conversation, so send these three queries in order in the same chat:

1. `I'm craving something Italian for two, and I already have eggs and parmesan at home. What would it cost?`
2. `Let's make the first one. What else can I cook with the leftovers?`
3. `What wine goes with it, and how is my week looking so far?`

Every tool the agent uses appears in the chat as a receipt strip. Click a strip to see exactly
what it sent and what came back.

"Start a new week" in the top right clears the conversation and your kitchen.


## Notes and limitations

- The free plan for the [wine API](https://wineapi.io/docs/) allows 100 calls per day, so a wine pairing request may fail if the limit is reached.
- Prices are estimates from a subset of a [Trader Joe's price list](https://traderjoesprices.com/) which does not have every ingredient in all the recipes. Also, measures like "2 cloves" can't be converted, so those lines assume one whole package and are marked low confidence.
- The recipe database doesn't say how many people a recipe serves, so amounts are scaled from an assumed 4 servings to the party size you ask for.
- Only full recipes can be added to your cart, not single ingredients.
- Pantry matching is by word: "rice" covers "Basmati Rice", but "chicken" would also cover "Chicken Stock". The cost breakdown shows which of your items covered each ingredient. Each owned item is assumed to cover up to 2 more meals (one use per committed meal) unless you say otherwise; after that it is priced normally.
- The wine database searches by wine name, winery, and appellation (not by food), so the agent picks a style and the tool verifies it against each wine's pairing list.
- Sessions live in server memory, so a restart starts fresh conversations.