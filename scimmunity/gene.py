def parse_gene_biotype(gtf, use_version=False):
    # adapted from kallisto t2g.py
    r = {}
    for line in gtf:
        if len(line) == 0 or line[0] == '#':
            continue
        l = line.strip().split('\t')
        if l[2] == 'gene':
            info = l[8]
            d = {}
            # get info in quotation marks for attribute
            for x in info.split('; '):
                x = x.strip()
                p = x.find(' ')
                if p == -1:
                    continue
                k = x[:p]
                p = x.find('"',p)
                p2 = x.find('"',p+1)
                v = x[p+1:p2] 
                d[k] = v


            if 'gene_id' not in d:
                continue
            
            if 'gene_biotype' not in d:
                continue

            gid = d['gene_id'].split(".")[0]
            if use_version:
                if 'gene_version' not in d:
                    continue
                gid += '.' + d['gene_version']
            
            if gid in r and r[gid] !=  (d['gene_name'], d['gene_biotype']):
                raise ValueError('{} {} {}'.format(gid, r[gid][1], d['gene_biotype'] ))
            else:
                r[gid] = (d['gene_name'], d['gene_biotype'])
    return r

def gene_biotype_dict(gtffile, prog='cellranger'):
    gtf = open(gtffile, 'r')
    r = parse_gene_biotype(gtf)
    return r

def get_gid2biotype(gtffile, prog='cellranger'):
    r = gene_biotype_dict(gtffile, prog=prog)
    return {gid:r[gid][1] for gid in r}
