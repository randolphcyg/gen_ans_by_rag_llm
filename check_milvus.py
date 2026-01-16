from pymilvus import connections, Collection

connections.connect(host="localhost", port="19530")
collection_name = "Vector_index_ec367307_db47_4449_9624_6e8ae9d6c405_Node"


if __name__ == '__main__':
    try:
        col = Collection(collection_name)
        print(f"\n=== 集合 {collection_name} 的 Schema ===")
        for field in col.schema.fields:
            print(f"字段名: {field.name:<20} | 类型: {field.dtype} | 描述: {field.description}")
            if field.dtype == 101: # FloatVector
                print(f"   -> 向量维度: {field.params}")
    except Exception as e:
        print(f"读取 Schema 失败: {e}")