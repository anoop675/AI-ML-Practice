# ai_ml_dl_projects
Repo for all implementations of AI/ML/DL algorithms and samples

ID3 algorithm:

function ID3(Examples, Target_Attribute, Attributes):    
    // Base Case 1: All examples have same class
    if all examples have same value for Target_Attribute:
        return a leaf node with that class label
    // Base Case 2: No more attributes to split
    if Attributes is empty:
        return a leaf node with majority class label
    // Recursive Case: Find best attribute and split
    else:
        // Step 1: Calculate Information Gain for each attribute
        for each attribute A in Attributes:
            IG(A) = InformationGain(Examples, A)
        // Step 2: Select attribute with highest Information Gain
        Best_Attribute = attribute with maximum IG
        // Step 3: Create decision node for Best_Attribute
        Tree = new decision tree node with Best_Attribute as root
        // Step 4: Split data and recursively build subtrees
        for each value v of Best_Attribute:
            // Get subset of examples where Best_Attribute = v
            Examples_v = subset of Examples where Best_Attribute has value v
            if Examples_v is empty:
                // Add leaf with majority class from Examples
                add leaf node with majority class of Examples
            else:
                // Recursively build subtree
                Remaining_Attributes = Attributes - {Best_Attribute}
                Subtree = ID3(Examples_v, Target_Attribute, Remaining_Attributes)
                add branch from Tree to Subtree with label v
