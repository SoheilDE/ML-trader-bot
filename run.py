import time
from trader_bot import main

if __name__ == "__main__":
    while True:
        main()
        time.sleep(60)  # Fetch data and make trading decisions every 1 minute