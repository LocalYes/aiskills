def flatten_batch(batch):
    lengths = [len(sublist) for sublist in batch]
    flat = [item for sublist in batch for item in sublist]
    return lengths, flat

def unflatten_batch(flat, lengths):
    reconstructed = []
    i = 0
    for length in lengths:
        reconstructed.append(flat[i:i+length])
        i += length
    return reconstructed
