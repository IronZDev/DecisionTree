#
# Intelligent Systems
# Deusto University
# Group I
# Maciej Stokfisz, Karolina Palka, 2019
#

library(shiny)
library(shinyjs)
library(shinyalert)
library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  useShinyjs(), # Using shinyJS for disabling and enabling elements
  useShinyalert(), # For pretty alerts
  
  # Application title
  titlePanel("Decision tree"),
  
  # Sidebar 
  sidebarLayout(
    sidebarPanel(
      # Upload box
      fileInput("file", 
                label = h3("Provide .csv file"),
                multiple = FALSE,
                accept = c(".csv")
      ),
      # Select category
      disabled(
        div(id = "controls",
          numericInput("minsplit", "Min split:", 500, min = 1, max = 1000, step = 1),
          numericInput("maxdepth", "Max depth:", 5, min = 1, max = 30, step = 1),
          numericInput("cp", "Cp:", 0.01, min = 0, max = 1, step = 0.01),
          actionButton("startBtn", "Calculate!")
        )
      )
    ),
    
    # Show calculated mean absolute error + graph
    mainPanel(
      h3(tags$b(textOutput("acc_test")), align="center"),
      plotOutput("acc_testPlot"),
      h3(tags$b(textOutput("acc_training")), align="center"),
      plotOutput("acc_trainingPlot")
    )
  ),
  h3(tags$b("Last decision tree obtained: "),align="center"),
  plotOutput("tree") 
)

server <- function(input, output, session) {
  data = reactiveVal(0) # Global variable for content of csv
  
  # Updating UI
  observeEvent(input$file, {
    # If not input file then disable category select and start button
    if (is.null(input$file)) {
      disable("controls")
      return(NULL)
    }
    
    
    # Check if file has .csv extension
    extension = strsplit(input$file$name, "\\.")
    if (extension[[1]][2] != "csv") {
      reset("file")
      reset("controls")
      disable("controls")
      shinyalert("Wrong file format!", type="error")
      return(NULL)
    }
    
    content = read.csv(file = input$file$datapath, header = TRUE, sep=",")
    colnames(content) <- gsub(".", " ", colnames(content), fixed=TRUE) # Replace "." with " " in column names
    content["phone number"] = NULL # Delete phone number
    data(content) # Save to reactive variable
    enable("controls")
  })
  
  checkBoundaries = function(id, name, minVal, maxVal) {
    if (isTruthy(input[[id]]) && input[[id]] < minVal) {
      updateNumericInput(session, id, value = minVal)
      shinyalert(paste0("Value of ", name, " cannot be lower than ", minVal, "!"), type="warning")
    }
    else if (isTruthy(input[[id]]) && input[[id]] > maxVal) {
      updateNumericInput(session, id, value = maxVal)
      shinyalert(paste0("Value of ", name, " cannot be higher than ", maxVal, "!"), type="warning")
    }
  }
  
  # Don't allow to exceed the boundaries
  observeEvent(input$minsplit, checkBoundaries("minsplit", "min split", 1, 1000))
  observeEvent(input$maxdepth, checkBoundaries("maxdepth", "max depth", 1, 30))
  observeEvent(input$cp, checkBoundaries("cp", "cp", 0, 1))
  
  checkIfEmpty = function(id, name) {
    if(!isTruthy(input[[id]])) {
      reset(id)
      shinyalert(paste0(name, " cannot be empty and must be a number!"), type="error")
      return(FALSE)
    }
    return(TRUE)
  }
  
  # Calcualte accuracy
  calculate_accuracy = function(real_val, predicted_val) {
    tab = table(real_val, predicted_val)
    return(sum(diag(tab)) / sum(tab))
  }
  
  # Start button clicked event
  observeEvent(input$startBtn, {
    # Check if values of inputs are empty or are not numbers
    if (!checkIfEmpty("minsplit", "Min split") ||
        !checkIfEmpty("maxdepth", "Max depth") ||
        !checkIfEmpty("cp", "Cp"))
      return()
    
    # Generate indices for partitioning
    indices = createMultiFolds(y=data()[["churn"]],5,1)
    accuracies_test = vector() # Vector of accuracies for test
    accuracies_training = vector() # Vector of accuracies for test
    for (index in  1:length(indices)) {
      # Do partitioning
      training.data = data()[indices[[index]], ] 
      test.data = data()[-indices[[index]], ]
      target = training.data[["churn"]]
      training.data[["churn"]] = NULL # Delete the column to be predicted from taining data
      decisionTree = rpart(target~., training.data, control = rpart.control(minsplit = input$minsplit, cp = input$cp, maxdepth = input$maxdepth))
      predictions_test = predict(decisionTree, test.data, type="class")
      predictions_training = predict(decisionTree, training.data, type="class")
      # Calculate confusion table and accuracy for test
      accuracy_test = calculate_accuracy(test.data$churn, predictions_test)
      accuracies_test = c(accuracies_test, accuracy_test)
      # Calculate confusion table and accuracy for training
      accuracy_training = calculate_accuracy(target, predictions_training)
      accuracies_training = c(accuracies_training, accuracy_training)
    }

    output$acc_test = renderText({
      return(paste0("Average accuracy of prediction based on test set: ",mean(accuracies_test)))
    })
    
    output$acc_testPlot = renderPlot({
      plotData = data.frame(num = 1:length(accuracies_test), accuracy_test = accuracies_test)
      ggplot(plotData, aes(x=num, y=accuracy_test))+
        geom_bar(stat="identity", fill="steelblue")+
        geom_text(aes(label=sprintf("%0.6f", round(accuracy_test, digits = 6))), position = position_stack(vjust = 0.5), color="white", size=5)+
        xlab("Number of fold as test data") + ylab("Accuracy value")+
        ggtitle("Prediction accuracy for particular folds based on test data")+
        theme(plot.title = element_text(hjust = 0.5, size=20, face="bold"),
              axis.title = element_text(size=15, face="bold"),
              axis.text = element_text(size=12))
    })
    
    output$acc_training = renderText({
      return(paste0("Average accuracy of prediction based on training set: ",mean(accuracies_training)))
    })

    output$acc_trainingPlot = renderPlot({
      plotData = data.frame(num = 1:length(accuracies_training), accuracy_training = accuracies_training)
      ggplot(plotData, aes(x=num, y=accuracy_training))+
        geom_bar(stat="identity", fill="blue")+
        geom_text(aes(label=sprintf("%0.6f", round(accuracy_training, digits = 6))), position = position_stack(vjust = 0.5), color="white", size=5)+
        xlab("Number of fold as test data") + ylab("Accuracy value")+
        ggtitle("Prediction accuracy for particular folds based on training data")+
        theme(plot.title = element_text(hjust = 0.5, size=20, face="bold"),
              axis.title = element_text(size=15, face="bold"),
              axis.text = element_text(size=12))
    })
    
    output$tree = renderPlot({
      rpart.plot(decisionTree)
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)

