resource "aws_ecs_cluster" "iris_cluster" {
  name = "iris-api-cluster"
}

resource "aws_ecs_task_definition" "iris_task" {
  family                   = "iris-api-task"
  requires_compatibilities  = ["FARGATE"]
  network_mode              = "awsvpc"
  cpu                       = "512"
  memory                    = "1024"

  container_definitions = jsonencode([
    {
      name      = "iris-api"
      image     = "353671347542.dkr.ecr.eu-west-1.amazonaws.com/iris-api:latest"
      essential = true
      portMappings = [
        {
          containerPort = 8080
          hostPort      = 8080
        }
      ]
      environment = [
        { name = "AWS_REGION", value = "eu-west-1" },
        { name = "MODEL_S3_URI", value = "s3://thebrowntiger/thesis-train-20250930-104353-2025-09-30-10-43-54-262/output/model.tar.gz" }
      ]
    }
  ])

  execution_role_arn = "arn:aws:iam::353671347542:role/ecsTaskExecutionRole"
}
