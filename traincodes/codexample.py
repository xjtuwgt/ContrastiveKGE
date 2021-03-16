from codex.codex import Codex

def codex_relation_analysis(codex_data: Codex):
    codex_relations = codex_data.relations()
    print('Number of relations = {}'.format(len(codex_relations)))
    for rel_id in list(codex_relations):
        rel_name = codex_data.relation_label(rel_id)
        rel_desc = codex_data.relation_description(rel_id)
        # print('{}:\t{}\n{}'.format(rel_id, rel_name, rel_desc))
        # print('-' * 75)

def codex_entity_analysis(codex_data: Codex):
    codex_entities = codex_data.entities()
    print('Number of entities = {}'.format(len(codex_entities)))
    for idx, ent_id in enumerate(list(codex_entities)):
        ent_name = codex_data.entity_label(ent_id)
        ent_desc = codex_data.entity_description(ent_id)
        ent_types = codex_data.entity_types(ent_id)
        ent_url = codex_data.entity_wikipedia_url(ent_id)
        # ent_type_desc = codex_data.entity_type_description(ent_id)
        print('{} - {}:\t{}\ndesc: {}\nent_type: {}\nurl: {}'.format(idx + 1, ent_id, ent_name, ent_desc, ent_types, ent_url))
        for ent_type_id in ent_types:
            type_url = codex_data.entity_type_wikipedia_url(ent_type_id)
            print('entity type {} = {}, {}'.format(ent_type_id, codex_data.entity_type_description(ent_type_id), type_url))
        print('-' * 75)

def codex_triple_analysis(codex_data: Codex):
    codex_triples = codex_data.triples() ### concat train, valid, test
    print('Number of triples = {}'.format(codex_triples.shape))

if __name__ == '__main__':
    codex_data = Codex()
    # codex_relation_analysis(codex_data=codex_data)
    codex_entity_analysis(codex_data=codex_data)
    # codex_triple_analysis(codex_data=codex_data)