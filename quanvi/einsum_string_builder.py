from dataclasses import dataclass
import string
from typing import List

@dataclass
class EinsumStringBuilder:
    '''
    This file is used to generate einsum strings for different situations.
    '''
    def _new_pool(self) -> List[str]:
        return list(string.ascii_lowercase + string.ascii_uppercase)
    
    def first_core_loss(self, D: int, E: int, Nk: int, Num_gates: int) -> str:
        pool = self._new_pool()
        lstate = pool.pop(0)
        rstate = pool.pop(0)
        lc = [pool.pop(0) for _ in range(Nk-1)]
        rc = [pool.pop(0) for _ in range(Nk-1)]
        lcp = []
        rcp = []
        middle = [[pool.pop(0), pool.pop(0)] for _ in range(max(0, Nk - D))]
        left_middle  = [row[0] for row in middle]
        right_middle = [row[1] for row in middle]
        for _ in range(min(D,Nk)):
            lcp.append(pool.pop(0))
            rcp.append(pool.pop(0))
        share_mode = pool.pop(0)
        lcore_str = ''.join([lstate] + lc + lcp + left_middle)
        rcore_str = ''.join(rcp + right_middle + [rstate] + rc)
        for i in range(min(D,Nk)):
            lcp[i] = lcp[i] + share_mode
            rcp[i] = rcp[i] + share_mode
        if Num_gates == 1:  #last core
            lstate = ','.join([lstate] + lc)
            rstate = ','.join([rstate] + rc)
            out_str = ''
        elif Nk < D:   # followed with cp core
            out_str = ''.join(lc + rc + [share_mode])
        else:   #followed without cp core
            out_str = ''.join(lc + rc)
        middle_str = [''.join(pair) for pair in middle]
        if(Nk<=D): # without extra core
            einsum_str = f"{lstate},{lcore_str},{','.join(lcp)},{','.join(rcp)},{rcore_str},{rstate}->{out_str}"  
        else:   # with extra core
            einsum_str = f"{lstate},{lcore_str},{','.join(lcp)},{','.join(rcp)},{rcore_str},{rstate},{','.join(middle_str)}->{out_str}"
        return einsum_str
    
    def cp_loss(self, num_k: int, is_first: bool, is_cp_last: bool, is_last: bool) -> str:
        Nk = num_k - 1     # Nk is the number of connections to the previous/next core
        pool = self._new_pool()
        F_l = [pool.pop(0) for _ in range(Nk)] if not is_first else []     # F_l/r refers to connections to the previous core
        F_r = [pool.pop(0) for _ in range(Nk)] if not is_first else []
        C_l = [pool.pop(0) for _ in range(Nk)]                             # C_l/r refers to connections to the next core
        C_r = [pool.pop(0) for _ in range(Nk)]
        middle = pool.pop(0)    # middle refers mode from cp core(B / DB)
        
        F_str = ''.join(F_l + F_r + [middle]) if not is_first else ''       # F_str is the string for the former einsum output
        L_str = ''.join(C_l + F_l + [middle]) if not is_last else ''.join(F_l + [middle])      # L_str is the string for the left core
        R_str = ''.join(F_r + [middle] + C_r) if not is_last else ''.join(F_r + [middle])      # R_str is the string for the right core
        if is_last: 
            out_str = ''
        elif is_cp_last: 
            out_str = ''.join(C_l + C_r)
        else: 
            out_str = ''.join(C_l + C_r + [middle])
        einsum_str = f"{F_str},{L_str},{R_str}->{out_str}" if not is_first else f"{L_str},{R_str}->{out_str}"
        return einsum_str

    def extra_loss(self, num_k: int, is_last: bool) -> str:
        Nk = num_k -1
        pool = self._new_pool()
        F_l = [pool.pop(0) for _ in range(Nk)]
        F_r = [pool.pop(0) for _ in range(Nk)]
        C_l = [pool.pop(0) for _ in range(Nk)] if not is_last else []
        C_r = [pool.pop(0) for _ in range(Nk)] if not is_last else []
        middle_left = pool.pop(0)
        middle_right = pool.pop(0)

        F_str = ''.join(F_l + F_r)
        L_str = ''.join(C_l + F_l + [middle_left]) if not is_last else ''.join(F_l + [middle_left])
        R_str = ''.join(F_r +[middle_right] +  C_r) if not is_last else ''.join(F_r + [middle_right])
        M_str = ''.join([middle_left] + [middle_right])
        out_str = ''.join(C_l + C_r) if not is_last else ''
        einsum_str = f"{F_str},{L_str},{R_str},{M_str}->{out_str}"
        return einsum_str

    def first_core_inquire(self, D: int, E: int, Nk: int, Num_gates: int) -> str:
        pool = self._new_pool()
        lstate = pool.pop(0)
        rstate = pool.pop(0)
        lc = [pool.pop(0) for _ in range(Nk-1)]
        rc = [pool.pop(0) for _ in range(Nk-1)]
        lcp = []
        rcp = []
        middle = [[pool.pop(0), pool.pop(0)] for _ in range(max(0, Nk - D))]
        left_middle  = [row[0] for row in middle]
        right_middle = [row[1] for row in middle]
        for _ in range(min(D,Nk)):
            lcp.append(pool.pop(0))
            rcp.append(pool.pop(0))
        share_mode = pool.pop(0)
        lcore_str = ''.join([lstate] + lc + lcp + left_middle)
        rcore_str = ''.join(rcp + right_middle + [rstate] + rc)
        for i in range(min(D,Nk)):
            lcp[i] = lcp[i] + share_mode
            rcp[i] = rcp[i] + share_mode
        if Num_gates == 1:  #last core
            lstate = ','.join([lstate] + lc)
            rstate = ','.join([rstate] + rc)
            out_str = ''.join([share_mode])
        else:
            out_str = ''.join(lc + rc + [share_mode])
        middle_str = [''.join(pair) for pair in middle]
        if(Nk<=D): # without extra core
            einsum_str = f"{lstate},{lcore_str},{','.join(lcp)},{','.join(rcp)},{rcore_str},{rstate}->{out_str}"  
        else:   # with extra core
            einsum_str = f"{lstate},{lcore_str},{','.join(lcp)},{','.join(rcp)},{rcore_str},{rstate},{','.join(middle_str)}->{out_str}"
        return einsum_str
    
    def cp_inquire(self, num_k: int, is_first: bool, is_cp_last: bool, is_last: bool) -> str:
        Nk = num_k - 1     # Nk is the number of connections to the previous/next core
        pool = self._new_pool()
        F_l = [pool.pop(0) for _ in range(Nk)] if not is_first else []     # F_l/r refers to connections to the previous core
        F_r = [pool.pop(0) for _ in range(Nk)] if not is_first else []
        C_l = [pool.pop(0) for _ in range(Nk)]                             # C_l/r refers to connections to the next core
        C_r = [pool.pop(0) for _ in range(Nk)]
        middle = pool.pop(0)    # middle refers mode from cp core(B / DB)
        
        F_str = ''.join(F_l + F_r + [middle]) if not is_first else ''       # F_str is the string for the former einsum output
        L_str = ''.join(C_l + F_l + [middle]) if not is_last else ''.join(F_l + [middle])      # L_str is the string for the left core
        R_str = ''.join(F_r + [middle] + C_r) if not is_last else ''.join(F_r + [middle])      # R_str is the string for the right core
        if is_last:
            out_str = ''.join([middle])
        else: 
            out_str = ''.join(C_l + C_r + [middle])
        einsum_str = f"{F_str},{L_str},{R_str}->{out_str}" if not is_first else f"{L_str},{R_str}->{out_str}"
        return einsum_str

    def extra_inquire(self, num_k: int, is_last: bool) -> str:
        Nk = num_k -1
        pool = self._new_pool()
        F_l = [pool.pop(0) for _ in range(Nk)]
        F_r = [pool.pop(0) for _ in range(Nk)]
        C_l = [pool.pop(0) for _ in range(Nk)] if not is_last else []
        C_r = [pool.pop(0) for _ in range(Nk)] if not is_last else []
        middle_left = pool.pop(0)
        middle_right = pool.pop(0)
        share_mode = pool.pop(0)

        F_str = ''.join(F_l + F_r + [share_mode])
        L_str = ''.join(C_l + F_l + [middle_left]) if not is_last else ''.join(F_l + [middle_left])
        R_str = ''.join(F_r +[middle_right] +  C_r) if not is_last else ''.join(F_r + [middle_right])
        M_str = ''.join([middle_left] + [middle_right])
        out_str = ''.join(C_l + C_r + [share_mode]) if not is_last else ''.join([share_mode])
        einsum_str = f"{F_str},{L_str},{R_str},{M_str}->{out_str}"
        return einsum_str
    

    def first_core_inquire_new(self, D: int, E: int, Nk: int, Num_gates: int) -> str:
        pool = self._new_pool()
        lstate = pool.pop(0)
        rstate = pool.pop(0)
        lc = [pool.pop(0) for _ in range(Nk-1)]
        rc = [pool.pop(0) for _ in range(Nk-1)]
        lcp = []
        rcp = []
        middle = [[pool.pop(0), pool.pop(0)] for _ in range(max(0, Nk - D))]
        left_middle  = [row[0] for row in middle]
        right_middle = [row[1] for row in middle]
        for _ in range(min(D,Nk)):
            lcp.append(pool.pop(0))
            rcp.append(pool.pop(0))

        share_mode = pool.pop(0)
        
        lcore_str = ''.join([lstate] + lc + lcp + left_middle)
        rcore_str = ''.join(rcp + right_middle + [rstate] + rc)
        cp = []
        for i in range(min(D,Nk)):
            cp.append(lcp[i] + rcp[i] + share_mode)
        if Num_gates == 1:  #last core
            lstate = ','.join([lstate] + lc)
            rstate = ','.join([rstate] + rc)
            out_str = ''.join([share_mode])
        else:
            out_str = ''.join(lc + rc + [share_mode])
        middle_str = [''.join(pair) for pair in middle]
        if(Nk<=D): # without extra core
            einsum_str = f"{lstate},{lcore_str},{','.join(cp)},{rcore_str},{rstate}->{out_str}"  
        else:   # with extra core
            einsum_str = f"{lstate},{lcore_str},{','.join(cp)},{rcore_str},{rstate},{','.join(middle_str)}->{out_str}"
        return einsum_str
    
    def cp_inquire_new(self, num_k: int, is_first: bool, is_cp_last: bool, is_last: bool) -> str:
        Nk = num_k - 1     # Nk is the number of connections to the previous/next core
        pool = self._new_pool()
        Former_Left = [pool.pop(0) for _ in range(Nk)] if not is_first else []
        Former_Right = [pool.pop(0) for _ in range(Nk)] if not is_first else []
        Latter_Left = [pool.pop(0) for _ in range(Nk)]
        Latter_Right = [pool.pop(0) for _ in range(Nk)]
        middle_left = pool.pop(0)
        middle_right = pool.pop(0)
        share_mode = pool.pop(0)    # middle refers mode from cp core(B / DB)

        F_str = ''.join(Former_Left + Former_Right + [share_mode])       # F_str is the string for the former einsum output
        L_str = ''.join(Latter_Left + Former_Left + [middle_left]) if not is_last else ''.join(Former_Left + [middle_left])      # L_str is the string for the left core
        R_str = ''.join(Former_Right + [middle_right] + Latter_Right) if not is_last else ''.join(Former_Right + [middle_right])      # R_str is the string for the right core
        M_str = ''.join([middle_left] + [middle_right] + [share_mode])
        out_str = ''.join(Latter_Left + Latter_Right + [share_mode]) if not is_last else ''.join([share_mode])
        einsum_str = f"{F_str},{L_str},{R_str},{M_str}->{out_str}"
        return einsum_str

    def extra_inquire_new(self, num_k: int, is_last: bool) -> str:
        Nk = num_k -1
        pool = self._new_pool()
        Former_Left = [pool.pop(0) for _ in range(Nk)]
        Former_Right = [pool.pop(0) for _ in range(Nk)]
        Latter_Left = [pool.pop(0) for _ in range(Nk)] if not is_last else []
        Latter_Right = [pool.pop(0) for _ in range(Nk)] if not is_last else []
        middle_left = pool.pop(0)
        middle_right = pool.pop(0)
        share_mode = pool.pop(0) #shared_mode refers mode from cp core(B / DB)

        F_str = ''.join(Former_Left + Former_Right + [share_mode])
        L_str = ''.join(Latter_Left + Former_Left + [middle_left]) if not is_last else ''.join(Former_Left + [middle_left])
        R_str = ''.join(Former_Right + [middle_right] + Latter_Right) if not is_last else ''.join(Former_Right + [middle_right])
        M_str = ''.join([middle_left] + [middle_right])
        out_str = ''.join(Latter_Left + Latter_Right + [share_mode]) if not is_last else ''.join([share_mode])
        einsum_str = f"{F_str},{L_str},{R_str},{M_str}->{out_str}"
        return einsum_str
