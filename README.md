# 24. Sinh viên điều chỉnh các siêu tham số trong mô hình như sau: max_depth chạy từ 2 đến 10 và
# max_leaf_nodes chạy từ 2 đến 10. Sau đó, vẽ biểu đồ thể hiện sự thay đổi của độ đo accuracy. Từ
# đó đưa đến kết luận với siêu tham số điều chỉnh nào thì mô hình tốt nhất.

# Khởi tạo mảng để lưu độ chính xác
max_depth_values = np.arange(2, 11)  # max_depth từ 2 đến 10
max_leaf_nodes_values = np.arange(2, 11)  # max_leaf_nodes từ 2 đến 10
accuracy_matrix = np.zeros((len(max_depth_values), len(max_leaf_nodes_values)))

# Lặp qua tất cả các giá trị của max_depth và max_leaf_nodes
for i, max_depth in enumerate(max_depth_values):
    for j, max_leaf_nodes in enumerate(max_leaf_nodes_values):
        # Khởi tạo mô hình cây quyết định với các siêu tham số
        clf = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
        
        # Huấn luyện mô hình
        clf.fit(X_train, y_train)
        
        # Dự đoán trên tập kiểm tra và tính toán độ chính xác
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        # Lưu độ chính xác vào ma trận
        accuracy_matrix[i, j] = accuracy

# Vẽ biểu đồ đường
plt.figure(figsize=(10, 6))

# Vẽ độ chính xác theo từng max_depth cố định và thay đổi max_leaf_nodes
for i, max_depth in enumerate(max_depth_values):
    plt.plot(max_leaf_nodes_values, accuracy_matrix[i, :], marker='o', label=f'Depth = {max_depth}')

# Thêm các chi tiết cho biểu đồ
plt.title('Độ chính xác của mô hình Decision Tree theo max_leaf_nodes với từng max_depth')
plt.xlabel('max_leaf_nodes')
plt.ylabel('Accuracy')
plt.legend(title='max_depth')
plt.grid(True)

# Hiển thị biểu đồ
plt.show()

#Mo hinh tot nhat voi max_depth = 10 và max_leaf_node = 6 hoặc 7
