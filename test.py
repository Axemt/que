from what.store import DirectoryStore
from what.prompts import QUERY_TEMPLATE

db = DirectoryStore()

query = 'que le gusta a ella?'

q_tk = db.chunkify(query)
q_emb= db.make_embed(q_tk)


relevant = db.get_paired_by_index(q_emb)

print(db.tomes)
print('====RELEVANT=====')
print(relevant)


context = db.search(query, formatted_context=True)


query_prompt = QUERY_TEMPLATE.format(context=context, query=query)
print(query_prompt)