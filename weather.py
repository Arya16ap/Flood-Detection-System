import requests

API_KEY = "a290e7fc221ba76dcc56b5d52b70d655"  # Replace with your OpenWeather API key

def get_live_weather():
    """Fetches real-time weather data using OpenWeather API."""
    city = input("\n🏙️ Enter the city for real-time weather data: ").strip()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()

    if "main" in response:
        temperature = response["main"]["temp"]
        pressure = response["main"]["pressure"]
        humidity = response["main"]["humidity"]
        tide = "High" if temperature > 25 else "Low"

        print("\n✅ Real-Time Weather Data Retrieved:")
        print(f"📍 City: {city}")
        print(f"🌡️ Temperature: {temperature}°C")
        print(f"🌪️ Pressure: {pressure} hPa")
        print(f"💧 Humidity: {humidity}%")
        print(f"🌊 Tide Level: {tide}\n")

        return {'Temperature': temperature, 'Pressure': pressure, 'Humidity': humidity, 'Tide': tide}
    
    print(f"⚠️ Error: Could not retrieve weather data for '{city}'. Switching to manual entry.")
    return get_user_weather()

def get_user_weather():
    """Gets storm-related values from user input."""
    print("\n🔹 Enter Storm Conditions Manually:")
    city = input("🏙️ Enter City Name: ").strip()
    temp = float(input("🌡️ Temperature (°C): "))
    pressure = float(input("🌪️ Pressure (hPa): "))
    humidity = float(input("💧 Humidity (%): "))
    tide = input("🌊 Tide Level (High/Low): ").strip().capitalize()

    return {'City': city, 'Temperature': temp, 'Pressure': pressure, 'Humidity': humidity, 'Tide': tide}

def get_weather_parameters():
    """Asks the user if they want real-time or manual storm data."""
    choice = input("\n📡 Do you want to use real-time weather data? (yes/no): ").strip().lower()
    return get_live_weather() if choice == "yes" else get_user_weather()
