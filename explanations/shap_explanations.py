import shap

# Create explainer
shap.initjs()

def global_shap(model, test_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data)

    # Verify shapes
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"X_test shape: {test_data.shape}")
    print(f"Expected value: {explainer.expected_value}")
    print(f"y_test shape: {test_data.shape}")
    print(f"Classes: {model.classes_}")

    # Test indexing for one class
    test_class_shap = shap_values[:, :, 0]
    print(f"Class 0 SHAP shape: {test_class_shap.shape}")
    print(f"Should be (97, 14): {test_class_shap.shape == (97, 14)}")
