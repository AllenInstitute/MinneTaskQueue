import caveclient
from taskqueue import queueable
import cloudfiles

client = caveclient.CAVEclient('minnie65_phase3_v1')

ct_df = client.materialize.query_table('aibs_metamodel_mtypes_v661_v2')

@queueable
def characterize_neuron(pt_root_id, cloud_path):
    print(f'characterizing {pt_root_id}')

    conn_df = client.materialize.query_view('connections_with_nuclei',
                                            filter_equal_dict={
                                                'pre_pt_root_id': pt_root_id,
                                            } )
    # merge the synapse table with the cell type table
    conn_dfm = conn_df.merge(ct_df[['pt_root_id','cell_type']],
                             left_on='post_pt_root_id',
                             right_on='pt_root_id', how='left')
    conn_dfm['cell_type'].fillna('unknown', inplace=True)
    # group conn_dfm by cell type and sum the number of synapses, and count rows
    conn_summary=conn_dfm.groupby('cell_type').agg({'sum_size':'sum', 'n_syn': 'sum', 'post_nuc_id': 'count'})
    conn_summary.rename(columns={'post_nuc_id':'n_connections'}, inplace=True)
    conn_summary['root_id']=pt_root_id


    # save the results
    cf= cloudfiles.CloudFiles(cloud_path)
    cf.put_json(f'{pt_root_id}.json', conn_summary.to_dict())
    return 
