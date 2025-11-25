docker build -t neural . && docker run -d --name neural_c -v ./source:/app neural && docker exec -it neural_c /bin/bash
