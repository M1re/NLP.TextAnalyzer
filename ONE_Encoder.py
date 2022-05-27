from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def ONE(tokens:[str]):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(tokens)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = OneHotEncoder(sparse=False)
    return onehot_encoder.fit_transform(integer_encoded)
