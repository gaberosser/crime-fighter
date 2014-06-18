## Functions that fetch data from the database
source('settings.R')
library('DBI')
library('RPostgreSQL')
library('rgeos')
drv <- dbDriver('PostgreSQL')

dbConnection <- function() {
  return(dbConnect(drv, dbname=DB_NAME))
}

getCamden <- function() {
  ## TODO: buffer the polygon to avoid issues with crimes near the boundary
  con <- dbConnection()
  res <- dbGetQuery(con, "SELECT ST_AsText(mpoly) FROM  database_division WHERE type_id='borough' AND name='Camden'")
  poly <- readWKT(res)
  dbDisconnect(con)
  return(poly)
}

getCad <- function(crime_type, dedupe=TRUE, only_new=FALSE) {
  NEW_CUTOFF_DATE = "2011-08-01"
  con <- dbConnection()
  crime_type <- paste(crime_type, collapse=", ")
  sql = "SELECT d.inc_datetime, ST_AsText(d.att_map) pt, d.cl01_id, d.cl02_id, d.cl03_id FROM database_cad d"
  if (dedupe) {
    sql <- c(sql, "JOIN (SELECT cris_entry, MIN(inc_datetime) md, MIN(id) mid FROM database_cad GROUP BY cris_entry) e ON d.cris_entry = e.cris_entry AND d.id = e.mid")
  }
  sql <- c(sql, "WHERE NOT (d.cris_entry ISNULL OR d.cris_entry LIKE 'NOT%')")
  sql <- c(sql, "AND NOT d.att_map ISNULL")
  sql <- c(sql, sprintf("AND (d.cl01_id IN (%1$s) OR d.cl02_id IN (%1$s) OR d.cl03_id IN (%1$s))", crime_type))
  if (only_new) {
    sql <- c(sql, sprintf("AND d.inc_datetime >= '%s'", NEW_CUTOFF_DATE))
  }
  sql <- c(sql, "ORDER BY d.inc_datetime")
  
  sql <- paste(sql, collapse=' ')
  res <- dbGetQuery(con, sql)
  dbDisconnect(con)
  res$pt <- as.array(apply(as.array(res$pt), MARGIN=1, FUN=readWKT))
  return(res)
}