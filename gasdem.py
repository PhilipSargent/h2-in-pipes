from zeep import Client

# Define the WSDL URL
wsdl_url = "https://data.nationalgas.com/apis/data-items?wsdl"
wsdl_url = "https://www.nationalgrid.com/MIPI"

# Create a SOAP client
client = Client(wsdl_url)

# Initialize an empty list to store the monthly gas demand data
monthly_gas_demand = []

# Loop over the months of 2020
for month in range(1, 13):
    # Call the SOAP method for the specific month
    response = client.service.GetMonthlyGasDemand(Year=2020, Month=month)

    # Check if the request was successful
    if response:
        # Append the monthly gas demand data to the list
        monthly_gas_demand.append(response)
    else:
        print(f"Failed to fetch data for month {month} of 2020")

# Print the monthly gas demand data
for month_data in monthly_gas_demand:
    print(month_data)
