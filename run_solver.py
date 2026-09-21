#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eğri Uzayzamanda Kolektif PE-NE Dinamiği: 2B Silindirik Sayısal Çözücü
Yazar: Mehmet İhsan Darende (Eylül 2026)
Referans: PE_NE_Dinamik_Doku_Hipotezi_Yayina_Hazir_v04
"""

import os
import glob
import numpy as np

# ----------------------------------------------------------------------
# 1. Dondurulmuş Evrensel ve Kuramsal Sabitler
# ----------------------------------------------------------------------
G_KPC = 4.30091e-6       # (km/s)^2 * kpc / M_sun
C_KMS = 299792.458       # km/s
A0_SI = 1.2e-10          # m/s^2
# 1 m/s^2 = 1e-3 km/s^2 = 1e-3 * (3.085677581e16 s / kpc)^(-1) vb.
# Birim çevrimi: 1 m/s^2 = 1.0e-3 * (1.0 / 3.085677581e16) (km/s)^2 / kpc * (saniye dönüşümü)
# Pratik: a0 (km^2 / (s^2 * kpc)) = A0_SI * (kpc_to_m) / (1000)^2
KPC_TO_M = 3.085677581e19
A0_KPC = A0_SI * (KPC_TO_M / 1.0e6)  # ~ 3.7028e3 (km/s)^2 / kpc

# Dondurulmuş Fotometrik Parametreler
UPSILON_D = 0.5
UPSILON_B = 0.7
ALPHA_CHI = 2.4e-3       # Cassini PPN üst sınırı
L_STAR = 2.0 * (ALPHA_CHI**3) * (C_KMS**2) / A0_KPC  # kpc (~0.7 kpc)

# ----------------------------------------------------------------------
# 2. Diferansiyel İletkenlik Fonksiyonu
# ----------------------------------------------------------------------
def mu_chi(y, eps=1e-5):
    """Analitik diferansiyel iletkenlik ve epsilon-regülarizasyonu."""
    y_reg = y + eps
    return y_reg / np.sqrt(1.0 + y_reg**2)

# ----------------------------------------------------------------------
# 3. İki Boyutlu Eksenel Simetrik Picard-SOR Çözücüsü
# ----------------------------------------------------------------------
def solve_chi_2d(R_nodes, rho_midplane, z0=0.4, NR=140, Nz=90, tol=1e-6, max_iter=2500, omega=1.25):
    """
    (R, z) silindirik koordinatlarında non-lineer PDE'yi çözer.
    Orta düzlem skaler gradyanını dchi/dR döndürür.
    """
    R_max = 2.5 * np.max(R_nodes)
    z_max = 1.5 * R_max
    
    r_grid = np.linspace(0.01, R_max, NR)
    z_grid = np.linspace(0.0, z_max, Nz)
    dr = r_grid[1] - r_grid[0]
    dz = z_grid[1] - z_grid[0]
    
    # 2B Baryonik yoğunluk matrisi: rho(R,z) = (Sigma(R)/(2*z0)) * sech^2(z/z0)
    R_2d, Z_2d = np.meshgrid(r_grid, z_grid, indexing='ij')
    Sigma_R = np.interp(r_grid, R_nodes, rho_midplane, left=rho_midplane[0], right=0.0)
    
    # Boyutsuz katsayı: 8 * pi * G / c^2 * alpha_chi
    source_coeff = (8.0 * np.pi * G_KPC / (C_KMS**2)) * ALPHA_CHI
    Source = np.zeros((NR, Nz))
    for i in range(NR):
        sech_term = 1.0 / np.cosh(np.clip(z_grid / z0, -50, 50))**2
        Source[i, :] = source_coeff * (Sigma_R[i] / (2.0 * z0)) * sech_term

    # Başlangıç alanı ve sınır koşulu
    M_bar_approx = np.trapz(2.0 * np.pi * r_grid * Sigma_R, r_grid)
    Q_chi = np.sqrt(2.0 * G_KPC * ALPHA_CHI * M_bar_approx / L_STAR) / C_KMS
    
    chi = np.zeros((NR, Nz))
    r_3d = np.sqrt(R_2d**2 + Z_2d**2)
    chi = Q_chi * np.log(np.maximum(r_3d / R_max, 1e-4))
    
    # Picard - SOR İterasyonu
    for it in range(max_iter):
        chi_old = chi.copy()
        
        # Gradyan normu y = l_star * sqrt((dchi/dr)^2 + (dchi/dz)^2)
        dchi_dr = np.gradient(chi, dr, axis=0)
        dchi_dz = np.gradient(chi, dz, axis=1)
        grad_norm = np.sqrt(dchi_dr**2 + dchi_dz**2)
        y_val = L_STAR * grad_norm
        mu = mu_chi(y_val)
        
        # İç noktaların güncellenmesi
        for i in range(1, NR-1):
            r_i = r_grid[i]
            for j in range(1, Nz-1):
                # Silindirik operatör ayrıklaştırması
                term_r = (mu[i, j] / dr**2) + (mu[i, j] / (2.0 * r_i * dr))
                term_z = mu[i, j] / dz**2
                diag = 2.0 * (mu[i, j] / dr**2 + mu[i, j] / dz**2)
                
                rhs = Source[i, j]
                val = (term_r * chi[i+1, j] + (term_r - mu[i, j]/(r_i*dr)) * chi[i-1, j] +
                       term_z * (chi[i, j+1] + chi[i, j-1]) - rhs) / diag
                
                chi[i, j] = (1.0 - omega) * chi[i, j] + omega * val
        
        # Sınır koşulları
        chi[0, :] = chi[1, :]          # R=0 simetri (dchi/dR = 0)
        chi[:, 0] = chi[:, 1]          # z=0 simetri (dchi/dz = 0)
        chi[-1, :] = Q_chi * np.log(np.sqrt(r_grid[-1]**2 + z_grid**2) / R_max)
        chi[:, -1] = Q_chi * np.log(np.sqrt(r_grid**2 + z_grid[-1]**2) / R_max)
        
        res = np.max(np.abs(chi - chi_old))
        if res < tol:
            break
            
    # Orta düzlemde (z=0) radyal gradyanı enterpole et
    dchi_dr_mid = np.gradient(chi[:, 0], dr)
    return np.interp(R_nodes, r_grid, dchi_dr_mid)

# ----------------------------------------------------------------------
# 4. Galaksi Çözümleme ve Tanısal Artık Değerlendirmesi
# ----------------------------------------------------------------------
def process_galaxy(dat_file):
    name = os.path.splitext(os.path.basename(dat_file))[0]
    # Sütunlar: R (kpc), v_obs (km/s), err_v (km/s), v_gas (km/s), v_disk (km/s), v_bul (km/s)
    data = np.loadtxt(dat_file, comments='#')
    R = data[:, 0]
    v_obs = data[:, 1]
    err_v = data[:, 2]
    v_gas = data[:, 3]
    v_disk = data[:, 4]
    v_bul = data[:, 5] if data.shape[1] > 5 else np.zeros_like(R)
    
    # Net baryonik dairesel hız
    v_bar_sq = (np.abs(v_gas)*v_gas + 
                UPSILON_D * np.abs(v_disk)*v_disk + 
                UPSILON_B * np.abs(v_bul)*v_bul)
    v_bar = np.sqrt(np.maximum(v_bar_sq, 0.0))
    
    # Kaba yüzey yoğunluğu tahmini: Sigma ~ v_bar^2 / (2 * pi * G * R)
    Sigma_approx = (v_bar**2) / (2.0 * np.pi * G_KPC * np.maximum(R, 0.1))
    
    # PDE Çözümü
    dchi_dR = solve_chi_2d(R, Sigma_approx)
    
    # Skaler hız: v_chi^2 = c^2 * alpha_chi * R * (dchi/dR)
    # Çekici dalda dchi/dR > 0
    v_chi_sq = (C_KMS**2) * ALPHA_CHI * R * np.maximum(dchi_dR, 0.0)
    v_chi = np.sqrt(v_chi_sq)
    
    # Model toplam hızı
    v_mod = np.sqrt(v_bar**2 + v_chi**2)
    
    # Tanısal kare artıklar
    residuals = (v_obs - v_mod) / err_v
    
    # R_split = 2.2 * R_d (Eğer R_d dosyada yoksa yaklaşık R_max / 3 alınır)
    R_split = 2.2 * (np.max(R) / 3.5)
    in_mask = R <= R_split
    out_mask = R > R_split
    
    xi2_tot = np.mean(residuals**2)
    xi2_in = np.mean(residuals[in_mask]**2) if np.any(in_mask) else 0.0
    xi2_out = np.mean(residuals[out_mask]**2) if np.any(out_mask) else 0.0
    
    return {
        'name': name,
        'N': len(R),
        'xi2_in': xi2_in,
        'xi2_out': xi2_out,
        'xi2_tot': xi2_tot,
        'R': R,
        'v_obs': v_obs,
        'err_v': err_v,
        'v_bar': v_bar,
        'v_chi': v_chi,
        'v_mod': v_mod,
        'res': residuals
    }

def main():
    data_dir = "data"
    files = sorted(glob.glob(os.path.join(data_dir, "*.dat")))
    if not files:
        print(f"Hata: '{data_dir}' dizininde .dat uzantılı girdi bulunamadı.")
        return

    print("================================================================================")
    print(f"{'Galaksi':<10} | {'N':<4} | {'xi^2_in':<10} | {'xi^2_out':<10} | {'xi^2_tot':<10}")
    print("--------------------------------------------------------------------------------")
    
    os.makedirs("outputs", exist_ok=True)
    
    for f in files:
        res = process_galaxy(f)
        print(f"{res['name']:<10} | {res['N']:<4} | {res['xi2_in']:<10.3f} | {res['xi2_out']:<10.3f} | {res['xi2_tot']:<10.3f}")
        
        # Noktasal CSV çıktısı
        out_csv = os.path.join("outputs", f"{res['name']}_results.csv")
        header = "R_kpc,v_obs_kms,err_v_kms,v_bar_kms,v_chi_kms,v_mod_kms,norm_residual"
        out_mat = np.column_stack([res['R'], res['v_obs'], res['err_v'], res['v_bar'], res['v_chi'], res['v_mod'], res['res']])
        np.savetxt(out_csv, out_mat, delimiter=",", header=header, comments="", fmt="%.3f")

    print("================================================================================")
    print("Bütün noktasal sonuç profilleri 'outputs/' dizinine kaydedildi.")

if __name__ == "__main__":
    main()
