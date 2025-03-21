const latInput = document.querySelector(".lat-input"); // Input field for latitude
const lonInput = document.querySelector(".lon-input"); // Input field for longitude
const searchButton = document.querySelector(".search-btn");
const locationOutput = document.querySelector(".location-output"); // Div for showing location

const LOCATION_KEY = "4f2f9351bb4603c4dc21c0bd094a2bb3"; // Replace with your actual OpenWeatherMap API key
const OPEN_METEO_API_URL = "https://archive-api.open-meteo.com/v1/archive";

// Function to get current date and 3 days before
const getDateRange = () => {
    const today = new Date();
    const endDate = today.toISOString().split('T')[0]; // Today's date
    const startDate = new Date(today.setDate(today.getDate() - 3)).toISOString().split('T')[0]; // 3 days before
    return { startDate, endDate };
};

// Function to get city and weather by coordinates
const getCityByCoordinates = async () => {
    const latitude = latInput.value.trim(); // Latitude
    const longitude = lonInput.value.trim(); // Longitude
    const { startDate, endDate } = getDateRange(); // Get start and end dates dynamically

    // Input validation for latitude and longitude
    if (!latitude || !longitude || isNaN(latitude)) {
        locationOutput.innerHTML = "<p>Please enter valid coordinates!</p>";
        return;
    }

    const REVERSE_GEOCODING_API_URL = `https://api.openweathermap.org/geo/1.0/reverse?lat=${latitude}&lon=${longitude}&limit=1&appid=${LOCATION_KEY}`;
    
    try {
        // Fetch data from reverse geocoding API
        const response = await fetch(REVERSE_GEOCODING_API_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        if (data.length > 0) {
            const { name, state, country } = data[0];
            locationOutput.innerHTML = `<h2>${name}, ${state}, ${country}</h2>`;
            
            // Fetch weather data from Open-Meteo API
            const weatherResponse = await fetch(`${OPEN_METEO_API_URL}?latitude=${latitude}&longitude=${longitude}&start_date=${startDate}&end_date=${endDate}&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,daylight_duration,precipitation_sum,shortwave_radiation_sum`);
            const weatherData = await weatherResponse.json();
            
            // Display weather details for each day separately
            let weatherHTML = "";
            for (let i = 0; i < 4; i++) { // Loop through each day (0 = today, 1 = day before, etc.)
                weatherHTML += `
                    <div class="weather-day">
                        <h3>Day ${i + 1} Weather Details</h3>
                        <p>Max Temperature: ${weatherData.daily.temperature_2m_max[i]} °C</p>
                        <p>Mean Temperature: ${weatherData.daily.temperature_2m_mean[i]} °C</p>
                        <p>Min Temperature: ${weatherData.daily.temperature_2m_min[i]} °C</p>
                        <p>Daylight Duration: ${weatherData.daily.daylight_duration[i]} seconds</p>
                        <p>Precipitation: ${weatherData.daily.precipitation_sum[i]} mm</p>
                    </div>
                `;
            }
            document.getElementById('weather-details').innerHTML = weatherHTML;

            // Send weather data to Flask for prediction
            const predictionResponse = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    latitude: latitude,
                    longitude: longitude,
                    weatherData: weatherData.daily
                })
            });
            
            const prediction = await predictionResponse.json();
            locationOutput.innerHTML += `<p>Predicted Streamflow: ${prediction.streamflow}</p>`;
        } else {
            locationOutput.innerHTML = "<p>Location not found!</p>";
        }
    } catch (error) {
        locationOutput.innerHTML = "<p>An error occurred while fetching the data.</p>";
        console.error("Error:", error);
    }
};

// Add event listener for search button
searchButton.addEventListener("click", getCityByCoordinates);
