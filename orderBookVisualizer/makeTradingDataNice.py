import json

with open('rawTradingData.json', 'r') as f:
    data = json.load(f)

csv_string = data['activitiesLog']

# The \n in the JSON string are literal \n characters, replace with real newlines
csv_string = csv_string.replace('\\n', '\n')

with open('tradingData.csv', 'w') as f:
    f.write(csv_string)

with open("tradingHistory.json", "w") as f:
    json.dump({"tradeHistory": data["tradeHistory"]}, f, indent=2)

print("Done — saved to tradingData.csv and tradingHistory.json")
