import os
import glob
import re

print("Starting conversion...")
directories = ["IEEE_TGRS_MDL-RS/MDL-RS_CNNs", "IEEE_TGRS_MDL-RS/MDL-RS_FC-Nets"]
for d in directories:
    for filepath in glob.glob(os.path.join(d, "*.py")):
        if "tf_utils" in filepath:
            continue
        print(f"Processing {filepath}...")
        with open(filepath, 'r') as f:
            content = f.read()
        
        # 1. replace tensorflow import
        if "import tensorflow.compat.v1" not in content:
            content = re.sub(
                r'import tensorflow as tf',
                'import os\nimport tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()',
                content
            )
        
        # 2. comment out tfdeterminism
        content = re.sub(r'(?m)^(\s*)from tfdeterminism import patch', r'\1# from tfdeterminism import patch', content)
        content = re.sub(r'(?m)^(\s*)patch\(\)', r'\1# patch()', content)
        
        # 3. replace tf.contrib.layers.variance_scaling_initializer
        content = re.sub(
            r'tf\.contrib\.layers\.variance_scaling_initializer\([^)]*\)',
            r'tf.keras.initializers.VarianceScaling(seed=1)',
            content
        )
        
        # 4. fix hardcoded save paths and ensure directory creation
        content = re.sub(
            r'["\']D:\\Python_Project\\MDL-RS/(save_[a-zA-Z0-9_-]+)/model\.ckpt["\']',
            r'"./\1/model.ckpt"',
            content
        )
        
        # Make directories dynamically prior to saving
        content = re.sub(
            r'(save_path = saver\.save\(sess,[ \t]*)(["\'][^"\']+["\'])(\))',
            r'os.makedirs(os.path.dirname(\2), exist_ok=True)\n        \1\2\3',
            content
        )
        
        # 5. Comment out plt.show()
        content = re.sub(r'plt\.show\(\)', r'# plt.show()', content)
        
        with open(filepath, 'w') as f:
            f.write(content)

print("Conversion complete.")
