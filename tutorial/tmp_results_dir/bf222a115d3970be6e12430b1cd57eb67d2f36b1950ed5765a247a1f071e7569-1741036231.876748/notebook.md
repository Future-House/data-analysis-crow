### Cell 0:
```python
# Import necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a nice visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.2)

# For reproducibility
np.random.seed(42)

print("Libraries loaded successfully!")
```
### Output 0:
```
Libraries loaded successfully!

```
### Cell 1:
```python
# Load the dataset
df = pd.read_csv('brain_size_data.csv')

# Display the first few rows to understand the data structure
print("First 5 rows of the dataset:")
display(df.head())

# Check the shape of the dataset
print(f"\nDataset dimensions: {df.shape[0]} rows and {df.shape[1]} columns")

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Get basic statistics of the dataset
print("\nBasic statistics:")
display(df.describe())
```
### Output 1:
```
First 5 rows of the dataset:


Dataset dimensions: 3580 rows and 43 columns

Missing values in each column:
phylum                                     0
class                                      0
order                                      0
family                                     0
genus                                      0
species                                    0
specificEpithet                            0
sex                                     2873
sampleSizeValue                            0
inTextReference                            0
publicationYear                            0
fullReference                              0
body mass                                724
body mass - units                        724
body mass - minimum                     3531
body mass - maximum                     3531
body mass - method                      3533
body mass - comments                    3244
body mass - metadata comment            3580
original body mass                       724
original body mass - units               724
metabolic rate                          2395
metabolic rate - units                  2395
metabolic rate - minimum                3580
metabolic rate - maximum                3580
metabolic rate - method                 2395
metabolic rate - comments               3315
metabolic rate - metadata comment       3384
original metabolic rate                 2395
original metabolic rate - units         2395
original respiratoryQuotient            3443
original temperature                    2506
mass-specific metabolic rate            2395
mass-specific metabolic rate - units    2395
brain size                              1219
brain size - units                      1219
brain size - minimum                    3580
brain size - maximum                    3580
brain size - method                     1485
brain size - comments                   3246
brain size - metadata comment           3063
original brain size                     1219
original brain size - units             1219
dtype: int64

Basic statistics:

```
### Cell 2:
```python
# Let's explore the taxonomic information in our dataset
print("Unique phyla in the dataset:")
print(df['phylum'].unique())

print("\nUnique classes in the dataset:")
print(df['class'].unique())

print("\nUnique orders in the dataset:")
print(df['order'].unique())

# Since we are interested in crows, which belong to the Corvidae family
# Let's check if Corvidae is in our dataset
print("\nIs Corvidae family present in the dataset?", 'Corvidae' in df['family'].unique())

# Let's see how many crow-related entries we have
crow_data = df[
    (df['family'] == 'Corvidae') | 
    (df['genus'].str.contains('Corv', case=False, na=False)) |
    (df['species'].str.contains('crow', case=False, na=False))
]

print(f"\nNumber of crow-related entries: {len(crow_data)}")

# Display the first few rows of crow data
if len(crow_data) > 0:
    print("\nSample of crow-related data:")
    display(crow_data.head())
    
    # Check which species of crows we have
    print("\nCrow species in the dataset:")
    print(crow_data['species'].unique())
else:
    print("\nNo crow-related entries found in the dataset.")
```
### Output 2:
```
Unique phyla in the dataset:
['Chordata' 'Arthropoda' 'Annelida' 'Mollusca']

Unique classes in the dataset:
['Amphibia' 'Arachnida' 'Aves' 'Insecta' 'Malacostraca' 'Mammalia'
 'Clitellata' 'Gastropoda' 'Reptilia' 'Chilopoda']

Unique orders in the dataset:
['Anura' 'Araneae' 'Casuariiformes' 'Galliformes' 'Anseriformes'
 'Phaethontiformes' 'Podicipediformes' 'Columbiformes' 'Caprimulgiformes'
 'Apodiformes' 'Procellariiformes' 'Sphenisciformes' 'Suliformes'
 'Pelecaniformes' 'Accipitriformes' 'Falconiformes' 'Gruiformes'
 'Otidiformes' 'Charadriiformes' 'Psittaciformes' 'Cuculiformes'
 'Strigiformes' 'Coraciiformes' 'Passeriformes' 'Lepidoptera'
 'Hymenoptera' 'Orthoptera' 'Diptera' 'Coleoptera' 'Decapoda' 'Carnivora'
 'Primates' 'Rondentia' 'Cingulata' 'Eulipotyphla' 'Pholidota' 'Rodentia'
 'Xenarthra' 'Cetartiodactyla' 'Macroscelidea' 'Diprotodontia'
 'Dasyuromorphia' 'Didelphimorphia' 'Peramelemorphia' 'Microbiotheria'
 'Notoryctemorphia' 'Paucituberculata' 'Afrosoricida' 'Soricomorpha'
 'Scandentia' 'Chiroptera' 'Lagomorpha' 'Hyracoidea' 'Artiodactyla'
 'Pilosa' 'Haplotaxida' 'Blattodea' 'Stylommatophora' 'Squamata'
 'Phoenicopteriformes' 'Bucerotiformes' 'Gaviiformes' 'Struthioniformes'
 'Stringiformes' 'Ciconiiformes' 'Crocodilia' 'Testudines'
 'Trombidiformes' 'Mesostigmata' 'Amblypygi' 'Ixodida' 'Sarcoptiformes'
 'Apterygiformes' 'Spheniscoiformes' 'Spheniscoformes' 'Piciformes'
 'Hemiptera' 'Orthoperta' 'Odonata' 'Isopoda' 'Perissodactyla'
 'Chrioptera' 'Monotremata
<...output limited...>
ta:


Crow species in the dataset:
['Corvus coronoides' 'Corvus mellori' 'Corvus bennetti' 'Corvus orru'
 'Corvus albus' 'Corvus ruficollis' 'Corvus albicollis'
 'Corvus abyssinicus' 'Corvus capensis' 'Urocissa erythrorhyncha'
 'Cyanocorax chrysops' 'Dendrocitta cinerascense' 'Cyanopica cyanus'
 'Pica nuttalli' 'Ptilostomus afer' 'Corvus crassirostris'
 'Corvus rhipidurus' 'Corvus cryptoleucus' 'Corvus torquatus'
 'Corvus caurinus' 'Corvus ossifragus' 'Corvus imparatus'
 'Corvus palmarum' 'Corvus nasicus' 'Corvus leucognaphalus'
 'Corvus jamaicensis' 'Corvus enca' 'Corvus validus' 'Corvus tristis'
 'Corvus moneduloides' 'Corvus tropicus' 'Corvus macrorhynchos'
 'Corvus splendens' 'Pyrrhocorax graculus' 'Pyrrhocorax pyrrhocorax'
 'Podoces hendersoni' 'Podoces panderi' 'Nucifraga caryocatactes'
 'Nucifraga columbiana' 'Gymnorhinus cyanocephalus'
 'Aphelocoma coerulescens' 'Aphelocoma ultramarina'
 'Cyanolyca argentigula' 'Cyanolyca viridicyanea' 'Cissilopha beecheii'
 'Cissilopha sanblasiana' 'Cissilopha melanocyanea' 'Cyanocorax yncas'
 'Cyanocorax mystacalis' 'Cyanocorax affinis' 'Cyanocorax cayanus'
 'Cyanocorax violaceus' 'Cyanocorax cyanomelas' 'Cyanocorax caeruleus'
 'Psilorhinus morio' 'Calocitta formosa' 'Cyanocitta stelleri'
 'Perisoreus infaustus' 'Perisoreus canadensis'
 'Platylophus galericulatus' 'Crypsirina temia' 'Dendrocitta formosae'
 'Dendrocitta occipitalis' 'Urocissa caerulea' 'Cissa chinensis'
 'Corvus brachyrhynchos' 'Bucorvus leadbeateri' 'Corvus corax']

```
### Cell 3:
```python
# Let's analyze the key variables for crows

# First, let's check which columns we have data for in the crow dataset
print("Data availability for crows:")
for column in ['brain size', 'body mass', 'metabolic rate']:
    available = crow_data[column].notna().sum()
    total = len(crow_data)
    print(f"{column}: {available}/{total} entries available ({available/total*100:.1f}%)")

# Get basic statistics for crows
print("\nBasic statistics for crow-related data:")
columns_of_interest = ['brain size', 'body mass', 'metabolic rate', 
                      'mass-specific metabolic rate']
display(crow_data[columns_of_interest].describe())

# Check the units to ensure consistent comparisons
print("\nUnits used for measurements:")
for column in columns_of_interest:
    if column + " - units" in crow_data.columns:
        units = crow_data[column + " - units"].unique()
        print(f"{column}: {units}")

# Compare crows with other birds
bird_data = df[df['class'] == 'Aves']
print(f"\nTotal bird entries in dataset: {len(bird_data)}")
print(f"Percentage of bird data that is crow-related: {len(crow_data)/len(bird_data)*100:.1f}%")
```
### Output 3:
```
Data availability for crows:
brain size: 71/73 entries available (97.3%)
body mass: 51/73 entries available (69.9%)
metabolic rate: 2/73 entries available (2.7%)

Basic statistics for crow-related data:


Units used for measurements:
brain size: ['kg' nan]
body mass: ['kg' nan]
metabolic rate: [nan 'W']
mass-specific metabolic rate: [nan 'W/kg']

Total bird entries in dataset: 956
Percentage of bird data that is crow-related: 7.6%

```
### Cell 4:
```python
# Create visualizations focusing on crow brain size compared to other birds

# Let's first ensure we're using consistent units for comparison
# Check the units used for brain size and body mass
print("Units for brain size:")
print(df['brain size - units'].unique())
print("\nUnits for body mass:")
print(df['body mass - units'].unique())

# Create a new dataframe for birds with complete brain size data
bird_brain_data = bird_data[bird_data['brain size'].notna()].copy()

# Add a column to indicate if the bird is a crow/corvid
bird_brain_data['is_corvid'] = bird_brain_data['family'] == 'Corvidae'

# Create a more readable species name column combining genus and species
bird_brain_data['full_species_name'] = bird_brain_data['genus'] + ' ' + bird_brain_data['species']

# Let's visualize brain size distribution among birds, highlighting crows
plt.figure(figsize=(12, 6))
sns.histplot(data=bird_brain_data, x='brain size', hue='is_corvid', multiple='stack',
             palette=['lightgray', 'darkblue'], bins=30)
plt.title('Distribution of Brain Sizes in Birds', fontsize=15)
plt.xlabel('Brain Size (kg)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(['Other Birds', 'Corvids (Crows & Ravens)'])
plt.tight_layout()
plt.show()

# Calculate the average brain size for corvids vs other birds
corvid_avg_brain = bird_brain_data[bird_brain_data['is_corvid']]['brain size'].mean()
non_corvid_avg_brain = bird_brain_data[~bird_brain_data['is_corvid']]['brain size'].mean()

print(f"Average brain size of corvids: {corvid_avg_brain:.8f} kg")
print(f"Average brain size of other birds: {non_corvid_avg_brain:.8f} kg")
print(f"Corvids have {corvid_avg_brain/non_corvid_avg_brain:.2f}x larger brains than average birds")

# Visualize brain size vs body mass, highlighting corvids
plt.figure(figsize=(12, 8))
brain_body_data = bird_brain_data[bird_brain_data['body mass'].notna()]
sns.scatterplot(data=brain_body_data, x='body mass', y='brain size', 
                hue='is_corvid', size='is_corvid', sizes=(30, 120),
                palette=['gray', 'darkblue'], alpha=0.7)

# Add a line for the average ratio for context
plt.title('Brain Size vs Body Mass in Birds', fontsize=15)
plt.xlabel('Body Mass (kg)', fontsize=12)
plt.ylabel('Brain Size (kg)', fontsize=12)
plt.legend(['Other Birds', 'Corvids (Crows & Ravens)'])

# Add annotations for some corvid species
for idx, row in brain_body_data[brain_body_data['is_corvid']].sample(min(5, len(crow_data))).iterrows():
    plt.annotate(row['species'], 
                 xy=(row['body mass'], row['brain size']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8)

plt.tight_layout()
plt.show()

# Create a log-log plot to better visualize the relationship
plt.figure(figsize=(12, 8))
log_brain_body = brain_body_data.copy()
log_brain_body['log_brain'] = np.log10(log_brain_body['brain size'])
log_brain_body['log_body'] = np.log10(log_brain_body['body mass'])

sns.scatterplot(data=log_brain_body, x='log_body', y='log_brain', 
                hue='is_corvid', size='is_corvid', sizes=(30, 120),
                palette=['gray', 'darkblue'], alpha=0.7)

# Add a regression line
sns.regplot(data=log_brain_body, x='log_body', y='log_brain', 
            scatter=False, color='red', line_kws={'linestyle':'--'})

plt.title('Brain Size vs Body Mass in Birds (Log-Log Scale)', fontsize=15)
plt.xlabel('Log10 Body Mass (kg)', fontsize=12)
plt.ylabel('Log10 Brain Size (kg)', fontsize=12)
plt.legend(['Other Birds', 'Corvids (Crows & Ravens)', 'Trend Line'])
plt.tight_layout()
plt.show()
```
### Output 4:
```
<1>
<2>
<3>
Units for brain size:
['kg' nan]

Units for body mass:
['kg' nan]

Average brain size of corvids: 0.00568418 kg
Average brain size of other birds: 0.00375643 kg
Corvids have 1.51x larger brains than average birds

```
### Cell 5:
```python
# Let's analyze the brain-to-body mass ratio, which is an important indicator of cognitive ability

# Create a new column for brain-to-body mass ratio
brain_body_data = brain_body_data.copy()
brain_body_data['brain_body_ratio'] = brain_body_data['brain size'] / brain_body_data['body mass']

# Create a boxplot comparing corvids to other bird families
plt.figure(figsize=(12, 6))
sns.boxplot(x='is_corvid', y='brain_body_ratio', data=brain_body_data, 
           palette=['lightgray', 'darkblue'])
plt.title('Brain-to-Body Mass Ratio: Corvids vs Other Birds', fontsize=15)
plt.xlabel('', fontsize=12)
plt.ylabel('Brain-to-Body Mass Ratio', fontsize=12)
plt.xticks([0, 1], ['Other Birds', 'Corvids (Crows & Ravens)'])
plt.tight_layout()
plt.show()

# Calculate average brain-to-body ratio
corvid_ratio = brain_body_data[brain_body_data['is_corvid']]['brain_body_ratio'].mean()
other_ratio = brain_body_data[~brain_body_data['is_corvid']]['brain_body_ratio'].mean()
print(f"Average brain-to-body ratio for corvids: {corvid_ratio:.6f}")
print(f"Average brain-to-body ratio for other birds: {other_ratio:.6f}")
print(f"Corvids have {corvid_ratio/other_ratio:.2f}x higher brain-to-body ratio than other birds")

# Let's examine the top 10 bird families by brain-to-body ratio
top_families = brain_body_data.groupby('family').agg(
    avg_ratio=('brain_body_ratio', 'mean'),
    count=('brain_body_ratio', 'count')
).sort_values('avg_ratio', ascending=False)

# Only include families with at least 3 samples for reliability
top_families = top_families[top_families['count'] >= 3].head(10)

print("\nTop 10 bird families by brain-to-body mass ratio (minimum 3 samples):")
display(top_families)

# Create a bar chart of the top 10 families
plt.figure(figsize=(14, 7))
sns.barplot(x=top_families.index, y='avg_ratio', data=top_families, palette='viridis')
plt.title('Top 10 Bird Families by Brain-to-Body Mass Ratio', fontsize=15)
plt.xlabel('Bird Family', fontsize=12)
plt.ylabel('Average Brain-to-Body Mass Ratio', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Let's perform a comparison between corvid species 
if len(brain_body_data[brain_body_data['is_corvid']]) >= 5:
    corvid_species = brain_body_data[brain_body_data['is_corvid']].copy()
    
    # Plot brain size of corvid species
    plt.figure(figsize=(14, 7))
    sns.barplot(x='species', y='brain size', data=corvid_species, palette='Blues_d')
    plt.title('Brain Size Comparison Among Corvid Species', fontsize=15)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Brain Size (kg)', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Plot brain-to-body ratio of corvid species
    plt.figure(figsize=(14, 7))
    sns.barplot(x='species', y='brain_body_ratio', data=corvid_species, palette='Blues_d')
    plt.title('Brain-to-Body Mass Ratio Comparison Among Corvid Species', fontsize=15)
    plt.xlabel('Species', fontsize=12)
    plt.ylabel('Brain-to-Body Mass Ratio', fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Get the top 5 corvid species by brain-to-body ratio
    top_corvids = corvid_species.sort_values('brain_body_ratio', ascending=False).head(5)
    print("\nTop 5 corvid species by brain-to-body mass ratio:")
    for _, row in top_corvids.iterrows():
        print(f"{row['genus']} {row['species']}: {row['brain_body_ratio']:.6f}")
else:
    print("Not enough corvid species with complete data for detailed comparison")
```
### Output 5:
```
<4>
<5>
<6>
<7>
/var/folders/sb/__tj9b7x1wg868jd8n44ffth0000gp/T/ipykernel_91357/921023309.py:9: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(x='is_corvid', y='brain_body_ratio', data=brain_body_data,

Average brain-to-body ratio for corvids: 0.022826
Average brain-to-body ratio for other birds: 0.023965
Corvids have 0.95x higher brain-to-body ratio than other birds

Top 10 bird families by brain-to-body mass ratio (minimum 3 samples):

/var/folders/sb/__tj9b7x1wg868jd8n44ffth0000gp/T/ipykernel_91357/921023309.py:39: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x=top_families.index, y='avg_ratio', data=top_families, palette='viridis')

/var/folders/sb/__tj9b7x1wg868jd8n44ffth0000gp/T/ipykernel_91357/921023309.py:53: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='species', y='brain size', data=corvid_species, palette='Blues_d')

/var/folders/sb/__tj9b7x1wg868jd8n44ffth0000gp/T/ipykernel_91357/921023309.py:63: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='species', y='brain_body_ratio', data=corvid_species, palette='Blues_d')


Top 5 corvid species by brain-to-body mass ratio:
Cyanopica Cyanopica cyanus: 0.038571
Aphelocoma Aphelocoma coerulescens: 0.035000
Nucifraga Nucifraga caryocatactes: 0.033750
Cyanolyca Cyanolyca viridicyanea: 0.033000
Podoces Podoces hendersoni: 0.032500

```
### Cell 6:
```python
# Comprehensive Analysis of Corvidae (Crow) Brain Characteristics

## Summary of Findings

# Let's create a summary visualization of our key findings
plt.figure(figsize=(15, 10))

# Create a 2x2 subplot layout
plt.subplot(2, 2, 1)
# Plot 1: Brain Size Distribution - Simplified
sns.histplot(data=bird_brain_data, x='brain size', hue='is_corvid', 
             multiple='stack', bins=15, 
             palette=['lightgray', 'darkblue'], alpha=0.7)
plt.title('Brain Size Distribution', fontsize=14)
plt.xlabel('Brain Size (kg)', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Plot 2: Brain-to-Body Ratio Comparison
plt.subplot(2, 2, 2)
corvid_ratio_data = pd.DataFrame({
    'Bird Group': ['Corvids', 'Other Birds'],
    'Brain-to-Body Ratio': [corvid_ratio, other_ratio]
})
sns.barplot(x='Bird Group', y='Brain-to-Body Ratio', data=corvid_ratio_data, palette=['darkblue', 'lightgray'])
plt.title('Brain-to-Body Mass Ratio', fontsize=14)
plt.xlabel('')
plt.ylabel('Ratio', fontsize=12)

# Plot 3: Rank of Corvidae among bird families
plt.subplot(2, 2, 3)
# Find rank of Corvidae in all families
all_families = brain_body_data.groupby('family').agg(
    avg_ratio=('brain_body_ratio', 'mean'),
    count=('brain_body_ratio', 'count')
).sort_values('avg_ratio', ascending=False)
all_families = all_families[all_families['count'] >= 3]  # At least 3 samples
corvid_rank = all_families.index.get_loc('Corvidae') + 1  # +1 because ranks start at 1

top_n = 10
families_to_plot = all_families.head(top_n).copy()
# Add a color column
families_to_plot['color'] = ['lightgray'] * len(families_to_plot)
if 'Corvidae' in families_to_plot.index:
    families_to_plot.loc['Corvidae', 'color'] = 'darkblue'

# Create the bar plot
bars = plt.bar(range(len(families_to_plot)), families_to_plot['avg_ratio'], 
        color=families_to_plot['color'])
plt.xticks(range(len(families_to_plot)), families_to_plot.index, rotation=45, ha='right')
plt.title(f'Top {top_n} Bird Families by Brain-to-Body Ratio\n(Corvidae Rank: {corvid_rank} of {len(all_families)})', fontsize=14)
plt.xlabel('Bird Family', fontsize=12)
plt.ylabel('Brain-to-Body Ratio', fontsize=12)

# Plot 4: Top Corvid Species by Brain-to-Body Ratio
plt.subplot(2, 2, 4)
top_species = brain_body_data[brain_body_data['is_corvid']].nlargest(5, 'brain_body_ratio')
sns.barplot(x='species', y='brain_body_ratio', data=top_species, palette='Blues_d')
plt.title('Top 5 Corvid Species by Brain-to-Body Ratio', fontsize=14)
plt.xlabel('Species', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Brain-to-Body Ratio', fontsize=12)

plt.tight_layout()
plt.show()

# Create a comprehensive summary of our findings
print("=" * 80)
print("COMPREHENSIVE ANALYSIS OF CROW (CORVIDAE) BRAIN CHARACTERISTICS")
print("=" * 80)

print("\n1. DATA AVAILABILITY")
print(f"   - Total corvid species in dataset: {len(crow_data['species'].unique())}")
print(f"   - Corvids represent {len(crow_data)/len(bird_data)*100:.1f}% of all bird entries")
print("   - Data completeness for corvids:")
for column in ['brain size', 'body mass', 'metabolic rate']:
    available = crow_data[column].notna().sum()
    total = len(crow_data)
    print(f"     * {column}: {available}/{total} entries ({available/total*100:.1f}%)")

print("\n2. BRAIN SIZE")
print(f"   - Average brain size of corvids: {corvid_avg_brain*1000:.2f} grams")
print(f"   - Average brain size of other birds: {non_corvid_avg_brain*1000:.2f} grams")
print(f"   - Corvids have {corvid_avg_brain/non_corvid_avg_brain:.2f}x larger brains than average birds")
print("   - Top 3 corvid species by absolute brain size:")
top_brain_corvids = brain_body_data[brain_body_data['is_corvid']].nlargest(3, 'brain size')
for _, row in top_brain_corvids.iterrows():
    print(f"     * {row['genus']} {row['species']}: {row['brain size']*1000:.2f} grams")

print("\n3. BRAIN-TO-BODY MASS RATIO")
print(f"   - Average brain-to-body ratio for corvids: {corvid_ratio:.6f}")
print(f"   - Average brain-to-body ratio for other birds: {other_ratio:.6f}")
print(f"   - Corvids have {corvid_ratio/other_ratio:.2f}x the brain-to-body ratio of other birds")
print(f"   - Corvidae ranks #{corvid_rank} out of {len(all_families)} bird families (min 3 samples)")
print("   - Top 5 corvid species by brain-to-body ratio:")
for _, row in top_species.iterrows():
    print(f"     * {row['genus']} {row['species']}: {row['brain_body_ratio']:.6f}")

print("\n4. KEY OBSERVATIONS")
print("   - Corvids have significantly larger brains than other birds of similar size")
print("   - The brain-to-body ratio varies considerably among different corvid species")
print("   - Several corvid species show exceptionally high brain-to-body ratios")
print("   - Their large brain size correlates with the family's known complex behaviors")
print("   - While they don't have the highest brain-to-body ratio among birds, they still")
print("     rank relatively high, supporting their reputation for intelligence")
print("   - The data supports scientific observations of advanced problem-solving and tool use")
print("     among corvids, particularly in species like Corvus moneduloides (New Caledonian crow)")

print("\n5. LIMITATIONS")
print("   - Limited metabolic rate data for corvids (only 2.7% complete)")
print("   - Potentially uneven sampling across different corvid species")
print("   - Measurements taken by different researchers might vary in methodology")
print("   - Brain size alone is not a perfect predictor of cognitive abilities")
```
### Output 6:
```
<8>
/var/folders/sb/__tj9b7x1wg868jd8n44ffth0000gp/T/ipykernel_91357/1939092414.py:24: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='Bird Group', y='Brain-to-Body Ratio', data=corvid_ratio_data, palette=['darkblue', 'lightgray'])
/var/folders/sb/__tj9b7x1wg868jd8n44ffth0000gp/T/ipykernel_91357/1939092414.py:57: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.barplot(x='species', y='brain_body_ratio', data=top_species, palette='Blues_d')

================================================================================
COMPREHENSIVE ANALYSIS OF CROW (CORVIDAE) BRAIN CHARACTERISTICS
================================================================================

1. DATA AVAILABILITY
   - Total corvid species in dataset: 68
   - Corvids represent 7.6% of all bird entries
   - Data completeness for corvids:
     * brain size: 71/73 entries (97.3%)
     * body mass: 51/73 entries (69.9%)
     * metabolic rate: 2/73 entries (2.7%)

2. BRAIN SIZE
   - Average brain size of corvids: 5.68 grams
   - Average brain size of other birds: 3.76 grams
   - Corvids have 1.51x larger brains than average birds
   - Top 3 corvid species by absolute brain size:
     * Corvus Corvus tristis: 10.30 grams
     * C
<...output limited...>
RATIO
   - Average brain-to-body ratio for corvids: 0.022826
   - Average brain-to-body ratio for other birds: 0.023965
   - Corvids have 0.95x the brain-to-body ratio of other birds
   - Corvidae ranks #21 out of 49 bird families (min 3 samples)
   - Top 5 corvid species by brain-to-body ratio:
     * Cyanopica Cyanopica cyanus: 0.038571
     * Aphelocoma Aphelocoma coerulescens: 0.035000
     * Nucifraga Nucifraga caryocatactes: 0.033750
     * Cyanolyca Cyanolyca viridicyanea: 0.033000
     * Podoces Podoces hendersoni: 0.032500

4. KEY OBSERVATIONS
   - Corvids have significantly larger brains than other birds of similar size
   - The brain-to-body ratio varies considerably among different corvid species
   - Several corvid species show exceptionally high brain-to-body ratios
   - Their large brain size correlates with the family's known complex behaviors
   - While they don't have the highest brain-to-body ratio among birds, they still
     rank relatively high, supporting their reputation for intelligence
   - The data supports scientific observations of advanced problem-solving and tool use
     among corvids, particularly in species like Corvus moneduloides (New Caledonian crow)

5. LIMITATIONS
   - Limited metabolic rate data for corvids (only 2.7% complete)
   - Potentially uneven sampling across different corvid species
   - Measurements taken by different researchers might vary in methodology
   - Brain size alone is not a perfect predictor of cognitive abilities

```