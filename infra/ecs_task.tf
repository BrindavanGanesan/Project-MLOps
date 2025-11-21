resource "aws_ecs_cluster" "iris_cluster" {
  name = "iris-api-cluster"
}

variable "model_s3_uri" {
  type        = string
  description = "S3 URI to the latest SageMaker-trained model"
}

resource "aws_ecs_task_definition" "iris_task" {
  family                   = "iris-api-task"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "512"
  memory                   = "1024"

  container_definitions = jsonencode([
    {
      name      = "iris-api"
      image     = "353671347542.dkr.ecr.eu-west-1.amazonaws.com/iris-api:v11"
      essential = true

      command = [
        "uvicorn",
        "app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8080"
      ]

      portMappings = [
        {
          containerPort = 8080
          hostPort      = 8080
          protocol      = "tcp"
        }
      ]

      environment = [
        { name = "AWS_REGION",       value = "eu-west-1" },
        { name = "MODEL_S3_URI",     value = var.model_s3_uri },
        { name = "PUSHGATEWAY_URL",  value = "http://108.130.158.94:9091" }
      ]

      logConfiguration = {
        logDriver = "awslogs",
        options = {
          "awslogs-group"         = "/ecs/iris-api-task"
          "awslogs-region"        = "eu-west-1"
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])

  execution_role_arn = "arn:aws:iam::353671347542:role/ecsTaskExecutionRole"
  task_role_arn      = "arn:aws:iam::353671347542:role/ecsTaskRole-iris-api"
}
