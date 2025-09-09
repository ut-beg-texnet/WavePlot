import numpy as np
    
def read_snp(m_filename = None,snp_filename = None): 
    # filename_snp = 'C:\Users\infla\OneDrive - University of Leeds\wave_model_files\3frac1\3frac1.snp';
# m_filename = 'C:\Users\infla\OneDrive - University of Leeds\wave_model_files\3frac1\3frac1.m';
    with open(snp_filename, 'rb') as f: 
        snap_1D =  np.fromfile(f, np.float32) 
        f.close()
    ##fid = open(snp_filename)
    #snap_1D = fread(fid,np.array([1,inf]),'single')
    #fid.close()
    parameters_snap,FD_parameters,lhis_parameters = scan_mfile_v2(m_filename)
    N_snap_type = parameters_snap.shape[1-1]
    grid = FD_parameters[1,1]
    dx_dz_dy = FD_parameters[3,1]
    nx = grid(1)
    
    nz = grid(2)
    
    ny = grid(3)
    
    cycles = FD_parameters[4]
    rep_vec = np.zeros((N_snap_type,1))
    for k in np.arange(1,N_snap_type+1).reshape(-1):
        rep_vec[k] = parameters_snap[k,2]
    
    rep = np.amax(rep_vec)
    if N_snap_type != 0:
        n_snap = cycles / rep
    else:
        n_snap = 0
    
    length_i = 0
    length_j = 0
    length_k = 0
    for m in np.arange(1,N_snap_type+1).reshape(-1):
        if 'vabs' == parameters_snap[m,1]:
            if 'i' == parameters_snap[m,3]:
                NZ_vabs_i = nz - 1
                NY_vabs_i = ny - 1
                NZ_vabs_i = NZ_vabs_i + 2
                length_vabs_i = NZ_vabs_i * NY_vabs_i
                length_i = length_i + length_vabs_i
            else:
                if 'j' == parameters_snap[m,3]:
                    NY_vabs_j = ny - 1
                    NX_vabs_j = nx - 1
                    NY_vabs_j = NY_vabs_j + 2
                    length_vabs_j = NY_vabs_j * NX_vabs_j
                    length_j = length_j + length_vabs_j
                else:
                    if 'k' == parameters_snap[m,3]:
                        NZ_vabs_k = nz - 1
                        NX_vabs_k = nx - 1
                        NZ_vabs_k = NZ_vabs_k + 2
                        length_vabs_k = NZ_vabs_k * NX_vabs_k
                        length_k = length_k + length_vabs_k
        else:
            if 'dil' == parameters_snap[m,1]:
                if 'i' == parameters_snap[m,3]:
                    NZ_dil_i = nz - 1
                    NY_dil_i = ny - 1
                    NZ_dil_i = NZ_dil_i + 2
                    length_dil_i = NZ_dil_i * NY_dil_i
                    length_i = length_i + length_dil_i
                else:
                    if 'j' == parameters_snap[m,3]:
                        NY_dil_j = ny - 1
                        NX_dil_j = nx - 1
                        NY_dil_j = NY_dil_j + 2
                        length_dil_j = NY_dil_j * NX_dil_j
                        length_j = length_j + length_dil_j
                    else:
                        if 'k' == parameters_snap[m,3]:
                            NZ_dil_k = nz - 1
                            NX_dil_k = nx - 1
                            NZ_dil_k = NZ_dil_k + 2
                            length_dil_k = NZ_dil_k * NX_dil_k
                            length_k = length_k + length_dil_k
            else:
                if 'xvel' == parameters_snap[m,1]:
                    if 'i' == parameters_snap[m,3]:
                        NZ_xvel_i = nz + 1
                        NY_xvel_i = ny + 1
                        NZ_xvel_i = NZ_xvel_i + 2
                        length_xvel_i = NZ_xvel_i * NY_xvel_i
                        length_i = length_i + length_xvel_i
                    else:
                        if 'j' == parameters_snap[m,3]:
                            NY_xvel_j = ny + 1
                            NX_xvel_j = nx + 1
                            NY_xvel_j = NY_xvel_j + 2
                            length_xvel_j = NY_xvel_j * NX_xvel_j
                            length_j = length_j + length_xvel_j
                        else:
                            if 'k' == parameters_snap[m,3]:
                                NZ_xvel_k = nz + 1
                                NX_xvel_k = nx + 1
                                NZ_xvel_k = NZ_xvel_k + 2
                                length_xvel_k = NZ_xvel_k * NX_xvel_k
                                length_k = length_k + length_xvel_k
                else:
                    if 'yvel' == parameters_snap[m,1]:
                        if 'i' == parameters_snap[m,3]:
                            NZ_yvel_i = nz + 1
                            NY_yvel_i = ny + 1
                            NZ_yvel_i = NZ_yvel_i + 2
                            length_yvel_i = NZ_yvel_i * NY_yvel_i
                            length_i = length_i + length_yvel_i
                        else:
                            if 'j' == parameters_snap[m,3]:
                                NY_yvel_j = ny + 1
                                NX_yvel_j = nx + 1
                                NY_yvel_j = NY_yvel_j + 2
                                length_yvel_j = NY_yvel_j * NX_yvel_j
                                length_j = length_j + length_yvel_j
                            else:
                                if 'k' == parameters_snap[m,3]:
                                    NZ_yvel_k = nz + 1
                                    NX_yvel_k = nx + 1
                                    NZ_yvel_k = NZ_yvel_k + 2
                                    length_yvel_k = NZ_yvel_k * NX_yvel_k
                                    length_k = length_k + length_yvel_k
                    else:
                        if 'zvel' == parameters_snap[m,1]:
                            if 'i' == parameters_snap[m,3]:
                                NZ_zvel_i = nz + 1
                                NY_zvel_i = ny + 1
                                NZ_zvel_i = NZ_zvel_i + 2
                                length_zvel_i = NZ_zvel_i * NY_zvel_i
                                length_i = length_i + length_zvel_i
                            else:
                                if 'j' == parameters_snap[m,3]:
                                    NY_zvel_j = ny + 1
                                    NX_zvel_j = nx + 1
                                    NY_zvel_j = NY_zvel_j + 2
                                    length_zvel_j = NY_zvel_j * NX_zvel_j
                                    length_j = length_j + length_zvel_j
                                else:
                                    if 'k' == parameters_snap[m,3]:
                                        NZ_zvel_k = nz + 1
                                        NX_zvel_k = nx + 1
                                        NZ_zvel_k = NZ_zvel_k + 2
                                        length_zvel_k = NZ_zvel_k * NX_zvel_k
                                        length_k = length_k + length_zvel_k
                        else:
                            if 's12' == parameters_snap[m,1]:
                                if 'i' == parameters_snap[m,3]:
                                    NZ_s12_i = nz + 1
                                    NY_s12_i = ny + 1
                                    NZ_s12_i = NZ_s12_i + 2
                                    length_s12_i = NZ_s12_i * NY_s12_i
                                    length_i = length_i + length_s12_i
                                else:
                                    if 'j' == parameters_snap[m,3]:
                                        NY_s12_j = ny + 1
                                        NX_s12_j = nx + 1
                                        NY_s12_j = NY_s12_j + 2
                                        length_s12_j = NY_s12_j * NX_s12_j
                                        length_j = length_j + length_s12_j
                                    else:
                                        if 'k' == parameters_snap[m,3]:
                                            NZ_s12_k = nz + 1
                                            NX_s12_k = nx + 1
                                            NZ_s12_k = NZ_s12_k + 2
                                            length_s12_k = NZ_s12_k * NX_s12_k
                                            length_k = length_k + length_s12_k
    
    tot_length = length_i + length_j + length_k
    #n_chunks = size(snap_1D,2)/tot_length;
    snaps = tuple(N_snap_type,n_snap)
    if tot_length != 0:
        snap_1D_snapshots = snap_1D(np.arange(1,tot_length * n_snap+1))
        snap_1_5D = np.reshape(snap_1D_snapshots, tuple(np.array([tot_length,n_snap])), order="F")
        n_samples = np.zeros((N_snap_type + 1,1))
        for h in np.arange(1,n_snap+1).reshape(-1):
            for m in np.arange(2,N_snap_type + 1+1).reshape(-1):
                if 'vabs' == parameters_snap[m - 1,1]:
                    if 'i' == parameters_snap[m - 1,3]:
                        n_samples[m] = length_vabs_i
                        snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_vabs_i,NY_vabs_i])), order="F")
                        snaps[m - 1,h] = snapshot(np.arange(2,-2),)
                    else:
                        if 'j' == parameters_snap[m - 1,3]:
                            n_samples[m] = length_vabs_j
                            snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NY_vabs_j,NX_vabs_j])), order="F")
                            snaps[m - 1,h] = snapshot[1:,]
                        else:
                            if 'k' == parameters_snap[m - 1,3]:
                                n_samples[m] = length_vabs_k
                                snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_vabs_k,NX_vabs_k])), order="F")
                                snaps[m - 1,h] = snapshot[1:,]
                else:
                    if 'dil' == parameters_snap[m - 1,1]:
                        if 'i' == parameters_snap[m - 1,3]:
                            n_samples[m] = length_dil_i
                            snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_dil_i,NY_dil_i])), order="F")
                            snaps[m - 1,h] = snapshot[1:,]
                        else:
                            if 'j' == parameters_snap[m - 1,3]:
                                n_samples[m] = length_dil_j
                                snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NY_dil_j,NX_dil_j])), order="F")
                                snaps[m - 1,h] = snapshot[1:,]
                            else:
                                if 'k' == parameters_snap[m - 1,3]:
                                    n_samples[m] = length_dil_k
                                    snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_dil_k,NX_dil_k])), order="F")
                                    snaps[m - 1,h] = snapshot[1:,]
                    else:
                        if 'xvel' == parameters_snap[m - 1,1]:
                            if 'i' == parameters_snap[m - 1,3]:
                                n_samples[m] = length_xvel_i
                                snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_xvel_i,NY_xvel_i])), order="F")
                                snaps[m - 1,h] = snapshot[1:,]
                            else:
                                if 'j' == parameters_snap[m - 1,3]:
                                    n_samples[m] = length_xvel_j
                                    snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NY_xvel_j,NX_xvel_j])), order="F")
                                    snaps[m - 1,h] = snapshot[1:,]
                                else:
                                    if 'k' == parameters_snap[m - 1,3]:
                                        n_samples[m] = length_xvel_k
                                        snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_xvel_k,NX_xvel_k])), order="F")
                                        snaps[m - 1,h] = snapshot[1:,]
                        else:
                            if 'yvel' == parameters_snap[m - 1,1]:
                                if 'i' == parameters_snap[m - 1,3]:
                                    n_samples[m] = length_yvel_i
                                    snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_yvel_i,NY_yvel_i])), order="F")
                                    snaps[m - 1,h] = snapshot[1:,]
                                else:
                                    if 'j' == parameters_snap[m - 1,3]:
                                        n_samples[m] = length_yvel_j
                                        snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NY_yvel_j,NX_yvel_j])), order="F")
                                        snaps[m - 1,h] = snapshot[1:,]
                                    else:
                                        if 'k' == parameters_snap[m - 1,3]:
                                            n_samples[m] = length_yvel_k
                                            snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_yvel_k,NX_yvel_k])), order="F")
                                            snaps[m - 1,h] = snapshot[1:,]
                            else:
                                if 'zvel' == parameters_snap[m - 1,1]:
                                    if 'i' == parameters_snap[m - 1,3]:
                                        n_samples[m] = length_zvel_i
                                        snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_zvel_i,NY_zvel_i])), order="F")
                                        snaps[m - 1,h] = snapshot[1:,]
                                    else:
                                        if 'j' == parameters_snap[m - 1,3]:
                                            n_samples[m] = length_zvel_j
                                            snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NY_zvel_j,NX_zvel_j])), order="F")
                                            snaps[m - 1,h] = snapshot[1:,]
                                        else:
                                            if 'k' == parameters_snap[m - 1,3]:
                                                n_samples[m] = length_zvel_k
                                                snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_zvel_k,NX_zvel_k])), order="F")
                                                snaps[m - 1,h] = snapshot[1:,]
                                else:
                                    if 's12' == parameters_snap[m - 1,1]:
                                        if 'i' == parameters_snap[m - 1,3]:
                                            n_samples[m] = length_s12_i
                                            snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_s12_i,NY_s12_i])), order="F")
                                            snaps[m - 1,h] = snapshot[1:,]
                                        else:
                                            if 'j' == parameters_snap[m - 1,3]:
                                                n_samples[m] = length_s12_j
                                                snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NY_s12_j,NX_s12_j])), order="F")
                                                snaps[m - 1,h] = snapshot[1:,]
                                            else:
                                                if 'k' == parameters_snap[m - 1,3]:
                                                    n_samples[m] = length_s12_k
                                                    snapshot = np.reshape(snap_1_5D(np.arange(sum(n_samples(np.arange(1,m - 1+1))) + 1,sum(n_samples(np.arange(1,m+1)))+1),h), tuple(np.array([NZ_s12_k,NX_s12_k])), order="F")
                                                    snaps[m - 1,h] = snapshot[1:,]
    
    dt = FD_parameters[2]
    n_rec_type = lhis_parameters.shape[1-1]
    lhis_1D = snap_1D(np.arange((tot_length * n_snap + 1),end()+1))
    length_lhis = 0
    
    tot_length_lhis = 0
    seismograms_2D = cell(1,n_rec_type)
    for n_r in np.arange(1,n_rec_type+1).reshape(-1):
        dt_lhis = lhis_parameters[n_r,2]
        dx_lhis = lhis_parameters[n_r,3]
        lhis_coord = lhis_parameters[n_r,4]
        lhis_i1 = lhis_coord(1,1)
        lhis_i2 = lhis_coord(1,2)
        n_rec_i = len(np.arange(lhis_i1,lhis_i2+dx_lhis,dx_lhis))
        lhis_j1 = lhis_coord(2,1)
        lhis_j2 = lhis_coord(2,2)
        n_rec_j = len(np.arange(lhis_j1,lhis_j2+dx_lhis,dx_lhis))
        lhis_k1 = lhis_coord(3,1)
        lhis_k2 = lhis_coord(3,2)
        n_rec_k = len(np.arange(lhis_k1,lhis_k2+dx_lhis,dx_lhis))
        rec_type = lhis_parameters[n_r,1]
        nt_lhis = int(np.floor(cycles / dt_lhis)) + 1
        if n_rec_i > 1:
            length_lhis = (n_rec_i + 2) * nt_lhis
            seisms_1D_i = lhis_1D(np.arange((tot_length_lhis + 1),(tot_length_lhis + length_lhis)+1))
            seisms_2D_i = np.reshape(seisms_1D_i, tuple(np.array([n_rec_i + 2,nt_lhis])), order="F")
            seisms_2D_i = seisms_2D_i[1:,]
            seismograms_2D[n_r] = np.transpose(seisms_2D_i)
        else:
            if n_rec_j > 1:
                length_lhis = (n_rec_j + 2) * nt_lhis
                seisms_1D_j = lhis_1D(np.arange((tot_length_lhis + 1),(tot_length_lhis + length_lhis)+1))
                seisms_2D_j = np.reshape(seisms_1D_j, tuple(np.array([n_rec_j + 2,nt_lhis])), order="F")
                seisms_2D_j = seisms_2D_j[1:,]
                seismograms_2D[n_r] = np.transpose(seisms_2D_j)
            else:
                length_lhis = (n_rec_k + 2) * nt_lhis
                seisms_1D_k = lhis_1D(np.arange((tot_length_lhis + 1),(tot_length_lhis + length_lhis)+1))
                seisms_2D_k = np.reshape(seisms_1D_k, tuple(np.array([n_rec_k + 2,nt_lhis])), order="F")
                seisms_2D_k = seisms_2D_k[1:,]
                seismograms_2D[n_r] = np.transpose(seisms_2D_k)
        tot_length_lhis = tot_length_lhis + length_lhis
        length_lhis = 0
        #     length_lhis_j = 0;
#     length_lhis_k = 0;
    
    return snaps,parameters_snap,FD_parameters,lhis_parameters,seismograms_2D

read_snp(m_filename = 'paper8.m',snp_filename = 'paper8.snp')
#read_hist_files(hst_filename = 'paper10.hst',filenameOut =  'paper10t.csv',cycles = 2000,dt = 1.87500e-03)










