"""
Script untuk visualisasi DTW Distance Matrix yang diperbaiki
Gunakan ini untuk mengganti cell visualisasi di notebook Anda

Cara penggunaan:
1. Jalankan semua cell di notebook sampai dtw_matrix terbentuk
2. Jalankan script ini dengan: %run visualize_dtw_improved.py
   atau copy-paste kode ini ke cell baru di notebook
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_dtw_matrix_improved(dtw_matrix, file_paths):
    """
    Visualisasi DTW distance matrix yang diperbaiki untuk dataset besar
    
    Parameters:
    -----------
    dtw_matrix : np.array
        Matriks DTW distance
    file_paths : list
        List path file audio
    """
    n_files = len(file_paths)
    file_names = [Path(fp).name for fp in file_paths]
    
    # Sesuaikan ukuran figure berdasarkan jumlah file
    fig_size = max(12, min(n_files * 0.3, 30))  # Min 12, Max 30
    plt.figure(figsize=(fig_size, fig_size))
    
    # Create heatmap
    im = plt.imshow(dtw_matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, label='DTW Distance', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    # Hanya tampilkan label untuk dataset kecil (< 20 files)
    if n_files < 20:
        plt.xticks(range(n_files), file_names, rotation=90, ha='right', fontsize=8)
        plt.yticks(range(n_files), file_names, fontsize=8)
        
        # Add values in cells hanya untuk matriks kecil
        if n_files < 10:
            for i in range(n_files):
                for j in range(n_files):
                    text_color = 'white' if dtw_matrix[i, j] > dtw_matrix.max() * 0.5 else 'black'
                    plt.text(j, i, f'{dtw_matrix[i, j]:.0f}',
                            ha="center", va="center", color=text_color, fontsize=7)
    else:
        # Untuk dataset besar, tampilkan label setiap N file
        step = max(1, n_files // 20)
        tick_indices = list(range(0, n_files, step))
        tick_labels = [file_names[i] for i in tick_indices]
        
        plt.xticks(tick_indices, tick_labels, rotation=90, ha='right', fontsize=7)
        plt.yticks(tick_indices, tick_labels, fontsize=7)
    
    plt.title(f'DTW Distance Matrix ({n_files} Audio Files)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Audio Files', fontsize=11, fontweight='bold')
    plt.ylabel('Audio Files', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print statistik DTW matrix
    print("\n" + "="*60)
    print("DTW DISTANCE MATRIX STATISTICS")
    print("="*60)
    # Ambil nilai non-diagonal (exclude diagonal yang bernilai 0)
    non_diag_values = dtw_matrix[~np.eye(n_files, dtype=bool)]
    print(f"Number of audio files: {n_files}")
    print(f"Total comparisons: {n_files * (n_files - 1) // 2}")
    print(f"Mean DTW distance: {np.mean(non_diag_values):.2f}")
    print(f"Median DTW distance: {np.median(non_diag_values):.2f}")
    print(f"Min DTW distance: {np.min(non_diag_values):.2f}")
    print(f"Max DTW distance: {np.max(non_diag_values):.2f}")
    print(f"Std deviation: {np.std(non_diag_values):.2f}")
    print("="*60)


# Jika script ini dijalankan langsung (bukan di-import)
if __name__ == "__main__":
    # Pastikan variabel dtw_matrix dan file_paths sudah ada
    try:
        visualize_dtw_matrix_improved(dtw_matrix, file_paths)
    except NameError:
        print("Error: Variabel 'dtw_matrix' dan 'file_paths' belum terdefinisi!")
        print("Jalankan cell-cell sebelumnya di notebook terlebih dahulu.")
