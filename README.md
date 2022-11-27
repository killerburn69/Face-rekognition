# Face-rekognition
<h1>Nhận diện gương mặt</h1>
<p>Folder data set là để chứa ảnh</p>
<p>Folder recognizer là để chứa training</p>
<p>People.csv chứa thông tin người nhập. Lưu ý chạy xong file csv thì vào file xóa dòng đầu vì dòng đầu trống</p>
<h1>Thứ tự chạy</h1>
<p>chạy file tung_sql_face để lấy data set, nó sẽ tự động chụp 100 tấm ảnh cắt theo frame</p>
<p>sau đó qua file chạy train_model để nó training và sinh ra file model ở trong folder rekognizer</p>
<p>Cuối cùng chạy file test_file</p>
<h3>Lưu ý khi chạy code nên tìm nguồn sáng tốt và có cam lap xịn thì dễ nhận diện hơn</h3>
<h1>Chạy trên streamlit. Chạy trên localhost</h1>
<p>chọn app.py và nhấn trên terminal của vscode là streamlit run app</p>
<p>Chạy từng step ghi trên app</p>
<p>Step 1: Điền id và tên. Sau đó nhấn nhất nhận vô csv. Rồi nhấn start và chờ video 30s sau đó stop</p>
<p>Step 2: Nhấn nút training. Thấy chữ running trên chạy xong r qua bước 3</p>
<p>Step 3: Test</p>
<h1>Lý do không chạy được trên deploy vì không truy cập camera trên deploy được và bị lỗi nên em chạy trên localhost</h1>
