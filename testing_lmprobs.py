import pickle
import random
import numpy as np
from lmprobs import TrigramSurprisalSpace

# Load the saved surprisal space object
with open('surprisals_8.pkl', 'rb') as f:
    tss = pickle.load(f)

print(f"Original size: {len(tss.surprisalvecs)}")

# Step 1: Initialize filtered structures
# Initially, filtered space is the full space
tss.currentsurprisalvecs = tss.surprisalvecs.copy()
# filtered_to_original maps filtered indices -> original indices
tss.filtered_to_original = list(range(len(tss.surprisalvecs)))

print(f"Filtered size before removal: {len(tss.currentsurprisalvecs)}")

# Step 2: Remove 1000 random indices from currentsurprisalvecs & update mapping
to_remove = random.sample(range(len(tss.currentsurprisalvecs)), 1000)
to_remove = sorted(to_remove, reverse=True)  # delete from highest index

for idx in to_remove:
    del tss.currentsurprisalvecs[idx]
    del tss.filtered_to_original[idx]

print(f"Filtered size after removal: {len(tss.currentsurprisalvecs)}")

# Step 3: Rebuild KDTree on filtered vectors
from sklearn.neighbors import KDTree
tss.nnfinder = KDTree(tss.currentsurprisalvecs)

# Step 4: Pick an index from original surprisalvecs to query
query_index = 5000  # change as needed
query_vec = tss.surprisalvecs[query_index].reshape(1, -1)

# Step 5: Query neighbors in filtered KDTree
distances, indices = tss.nnfinder.query(query_vec, k=5)
indices = indices[0]  # flatten

print(f"\nQuerying neighbors for original vector index {query_index}")
print(f"Returned indices (in filtered space): {indices}")

# Step 6: Check if neighbors match filtered vectors or original vectors at these indices
print("\nCheck neighbor vector matches:")
for i in indices:
    neighbor_vec = tss.currentsurprisalvecs[i]
    matches_filtered = np.allclose(neighbor_vec, tss.currentsurprisalvecs[i])
    matches_original_same_index = np.allclose(neighbor_vec, tss.surprisalvecs[i])
    # Check if neighbor_vec matches any vector in original surprisalvecs
    matches_any_original = any(np.allclose(neighbor_vec, v) for v in tss.surprisalvecs)
    
    print(f"Index in filtered: {i}")
    print(f"Matches filtered vector at i? {matches_filtered}")
    print(f"Matches original vector at same index? {matches_original_same_index}")
    print(f"Matches any original vector? {matches_any_original}")
    print(f"Original index of this neighbor: {tss.filtered_to_original[i]}")
    print("------")

# Step 7: Show mapping from filtered indices back to original indices
print("\nFiltered to original index mapping for neighbors:")
for i in indices:
    print(f"Filtered idx {i} -> Original idx {tss.filtered_to_original[i]}")
print("\nCheck neighbor vector matches (with actual vector values):")
for i in indices:
    neighbor_vec = tss.currentsurprisalvecs[i]
    original_index = tss.filtered_to_original[i]
    original_vec = tss.surprisalvecs[original_index]

    # Checks
    matches_filtered = np.allclose(neighbor_vec, tss.currentsurprisalvecs[i])
    matches_same_index_original = np.allclose(neighbor_vec, tss.surprisalvecs[i])
    matches_mapped_original = np.allclose(neighbor_vec, original_vec)
    matches_any_original = any(np.allclose(neighbor_vec, v) for v in tss.surprisalvecs)

    print(f"\nFiltered index: {i}")
    print(f"  Mapped original index: {original_index}")

    print("  → Filtered vector:")
    print(neighbor_vec)

    print("  → Original vector at same index (tss.surprisalvecs[i]):")
    print(tss.surprisalvecs[i])

    print("  → Original vector at mapped index (tss.surprisalvecs[original_index]):")
    print(original_vec)

    print(f"  Matches filtered vector at index i?           {'✅' if matches_filtered else '❌'}")
    print(f"  Matches original vector at index i?           {'✅' if matches_same_index_original else '❌'}")
    print(f"  Matches mapped original vector ({original_index})? {'✅' if matches_mapped_original else '❌'}")
    print(f"  Matches any original vector at all?           {'✅' if matches_any_original else '❌'}")
    print("------")
