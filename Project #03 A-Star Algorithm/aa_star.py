from dists import dists as d, straight_line_dists_from_bucharest
import pandas as pd
pd.options.mode.chained_assignment = None

df_dists = pd.DataFrame([[k,dist[0],dist[1]] for k in d for dist in d[k]], columns=['de','para','dist'])
df_dists['para_linha_reta'] = df_dists['para'].apply(lambda x: straight_line_dists_from_bucharest[x])
df_dists['acum'] = 0
df_dists['total'] = 0
df_dists['rota'] = [[] for _ in df_dists.values]

def a_star(start, goal='Bucharest'):
    borda = pd.DataFrame()
    borda = pd.concat([borda, df_dists[df_dists.de==start]])
    borda['total'] = borda['acum'] + borda['para_linha_reta']
    borda = borda.sort_values(by='total',ascending=True).reset_index(drop=True)
    borda['rota'] = [[start] for _ in borda.values]

    def interaction(borda):
        select = borda.iloc[0]
        result = select.rota
        borda = borda.iloc[1:]

        df_dists_temp = df_dists[df_dists.de==select.para]
        df_dists_temp['acum'] = select.acum + df_dists_temp['dist']
        df_dists_temp['rota'] = df_dists.apply(lambda x: select.rota + [x.de],axis=1)

        borda = pd.concat([borda, df_dists_temp])
        borda['total'] = borda['acum'] + borda['para_linha_reta']
        borda = borda.sort_values(by='total',ascending=True).reset_index(drop=True)

        if select.para != goal:
            borda, result = interaction(borda) 
        else:
            result+=[goal]

        return borda, result

    if start != goal:
        borda, result = interaction(borda) 
    else: 
        result = [start]

    return result

if __name__ == "__main__":
    for k in straight_line_dists_from_bucharest:
        rota = a_star(k)
        print(f'A melhor rota de {k} para Bucharest é:\n{rota}\n')