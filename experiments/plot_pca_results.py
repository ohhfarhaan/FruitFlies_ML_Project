import matplotlib.pyplot as plt

components = [10, 20, 40, 60, 80]

mae = [
    0.051617,
    0.045173,
    0.038482,
    0.033497,
    0.035308
]

rmse = [
    0.074170,
    0.066508,
    0.054732,
    0.050255,
    0.050207
]

r2 = [
    0.966889,
    0.973376,
    0.981969,
    0.984799,
    0.984828
]

plt.figure()
plt.plot(components, mae, marker='o')
plt.xlabel('PCA Components')
plt.ylabel('MAE (radians)')
plt.title('Wing-Angle MAE vs PCA Components')
plt.xticks(components)
plt.grid(True)
plt.savefig('output/graphs/pca_wing_mae.png', dpi=200)
plt.show()

plt.figure()
plt.plot(components, rmse, marker='o')
plt.xlabel('PCA Components')
plt.ylabel('RMSE (radians)')
plt.title('Wing-Angle RMSE vs PCA Components')
plt.xticks(components)
plt.grid(True)
plt.savefig('output/graphs/pca_wing_rmse.png', dpi=200)
plt.show()

plt.figure()
plt.plot(components, r2, marker='o')
plt.xlabel('PCA Components')
plt.ylabel('R²')
plt.title('Wing-Angle R² vs PCA Components')
plt.xticks(components)
plt.grid(True)
plt.savefig('output/graphs/pca_wing_r2.png', dpi=200)
plt.show()
