# Pantera-iGEM
This repository documents the mathematical models published during the Igem Design League 2023 competition by the mathematical modeling section of the iGEM Panther team.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install matplotlib and pandas.

```bash
pip install matplotlib
pip install pandas 
```

### Usage

```python
import matplotlib.pyplot as plt
import pandas as pd

# Convert an array or list to Data Frame
saves_ni_m = pd.DataFrame(ni_m)

# Save the Data Frame as a csv file
saves_ni_m.to_csv('Polinator.csv', index=False)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors

- Mauricio Silva Tovar
- Brian Rivera Martinez
- Laura Mariana Reyes Morales
- Dra. Helen Lugo Méndez
- Dr. Adrian Mauricio Escobar Ruiz
- Dr. Jose Luis del Rio Correa

## License

[MIT](https://choosealicense.com/licenses/mit/)
