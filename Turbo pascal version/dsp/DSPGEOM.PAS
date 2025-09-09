  {Part of Dspl Unit}
   {Geometry and grid routines}

    PROCEDURE Draw2D_Move (i1,j1: Integer);
      BEGIN
        WITH Scale2D DO
          MOVETO (Xoff+TRUNC(Xpx*(i1-XX1)) DIV Xlen, Yoff+TRUNC(Ypx*(j1-YY1)) DIV Ylen);
      END;


    PROCEDURE Draw2D_Rect (px1,py1,px2,py2 : LongInt);
      BEGIN
        WITH Scale2D DO
          RECTANGLE ( Xoff + (Xpx*(px1-XX1)) DIV Xlen, Yoff + (Ypx*(py1-YY1)) DIV Ylen,
                      Xoff + (Xpx*(px2-XX1)) DIV Xlen, Yoff + (Ypx*(py2-YY1)) DIV Ylen );
{         RECTANGLE ( Xoff + TRUNC(Xpx*(px1-0.125-XX1)) DIV Xlen, Yoff + TRUNC(Ypx*(py1-0.125-YY1)) DIV Ylen,
                Xoff + TRUNC(Xpx*(px2+0.125-XX1)) DIV Xlen, Yoff + TRUNC(Ypx*(py2+0.125-YY1)) DIV Ylen );}
{         RECTANGLE ( Xoff + TRUNC(Xpx*(px1-0.125-XX1)/ Xlen), Yoff + TRUNC(Ypx*(py1-0.125-YY1) / Ylen),
                Xoff + TRUNC(Xpx*(px2+0.125-XX1) / Xlen), Yoff + TRUNC(Ypx*(py2+0.125-YY1) / Ylen) );}
      END;

    PROCEDURE Draw2D_Bar (px1,py1,px2,py2 : Integer);
      BEGIN
        WITH Scale2D DO
          BAR ( Xoff + TRUNC(Xpx*(px1-0.5-XX1)) DIV Xlen, Yoff + TRUNC(Ypx*(py1-0.5-YY1)) DIV Ylen,
                Xoff + TRUNC(Xpx*(px2+0.5-XX1)) DIV Xlen, Yoff + TRUNC(Ypx*(py2+0.5-YY1)) DIV Ylen );
      END;

    PROCEDURE Draw2D_Line (i1,j1, i2,j2 : Integer);
      BEGIN
        WITH Scale2D DO
          LINE ( Xoff + (Xpx*(i1-XX1)) DIV Xlen, Yoff + (Ypx*(j1-YY1)) DIV Ylen,
                 Xoff + (Xpx*(i2-XX1)) DIV Xlen, Yoff + (Ypx*(j2-YY1)) DIV Ylen );
      END;

    PROCEDURE Draw2D_rLine (px1,py1,px2,py2 : single);
      BEGIN
        WITH Scale2D DO
          LINE ( Xoff + TRUNC(Xpx*(px1-XX1)/Xlen), Yoff + TRUNC(Ypx*(py1-YY1)/Ylen),
                 Xoff + TRUNC(Xpx*(px2-XX1)/Xlen), Yoff + TRUNC(Ypx*(py2-YY1)/Ylen)  );
      END;

    PROCEDURE Draw2D_Point (i1,j1: Integer;  s:string);
      BEGIN
        WITH Scale2D DO
          MOVETO (Xoff+TRUNC(Xpx*(i1-XX1)) DIV Xlen, Yoff+TRUNC(Ypx*(j1-YY1)) DIV Ylen);
        OutText ('');
        MOVEREL(-2,-4); IF (s <> ' ') THEN OutText (s);
      END;


    PROCEDURE Draw3D_Line (i1,j1,k1, i2,j2,k2 : Integer);
      VAR
        Vert : ARRAY [1..2,1..2] OF Integer;
        Xsc, Ysc : single;
      BEGIN
        WITH Scale2D DO
          BEGIN
{            k1 := k1-1; k2 := k2-1;}
            Xsc := Xpx/Xlen; Ysc := Ypx/Ylen;
            Vert[1,1] := Xoff + ROUND (Xsc* (i1*Rot[1,1]+j1*Rot[2,1]+k1*Rot[3,1] -XX1) );
            Vert[1,2] := Yoff + ROUND (Ysc* (i1*Rot[1,2]+j1*Rot[2,2]+k1*Rot[3,2] -YY1) );
            Vert[2,1] := Xoff + ROUND (Xsc* (i2*Rot[1,1]+j2*Rot[2,1]+k2*Rot[3,1] -XX1) );
            Vert[2,2] := Yoff + ROUND (Ysc* (i2*Rot[1,2]+j2*Rot[2,2]+k2*Rot[3,2] -YY1) );
            Line (Vert[1,1],Vert[1,2],Vert[2,1],Vert[2,2]);
          END;
      END; {Draw3D_Line}

    PROCEDURE Draw3D_rLine (px1,py1,pz1, px2,py2,pz2 : single);
      VAR
        Vert : ARRAY [1..2,1..2] OF Integer;
        Xsc, Ysc : single;
      BEGIN
        WITH Scale2D DO
          BEGIN
            Xsc := Xpx/Xlen; Ysc := Ypx/Ylen;
            Vert[1,1] := Xoff + ROUND (Xsc* (px1*Rot[1,1]+py1*Rot[2,1]+pz1*Rot[3,1] -XX1) );
            Vert[1,2] := Yoff + ROUND (Ysc* (px1*Rot[1,2]+py1*Rot[2,2]+pz1*Rot[3,2] -YY1) );
            Vert[2,1] := Xoff + ROUND (Xsc* (px2*Rot[1,1]+py2*Rot[2,1]+pz2*Rot[3,1] -XX1) );
            Vert[2,2] := Yoff + ROUND (Ysc* (px2*Rot[1,2]+py2*Rot[2,2]+pz2*Rot[3,2] -YY1) );
            Line (Vert[1,1],Vert[1,2],Vert[2,1],Vert[2,2]);
          END;
      END; {Draw3D_rLine}

    PROCEDURE Draw3D_Point (i1,j1,k1: Integer;  s:string);
      VAR  Vert : ARRAY [1..1,1..2] OF Integer;
      BEGIN
        WITH Scale2D DO
          BEGIN
            Vert[1,1] := Xoff + ROUND (Xpx* (i1*Rot[1,1]+j1*Rot[2,1]+k1*Rot[3,1] -XX1) / Xlen);
            Vert[1,2] := Yoff + ROUND (Ypx* (i1*Rot[1,2]+j1*Rot[2,2]+k1*Rot[3,2] -YY1) / Ylen);
          END;
        MoveTo (Vert[1,1],Vert[1,2]); OutText (''); {Alt-127}
        MOVEREL(-2,-4); IF (s <> ' ') THEN OutText (s);
      END;


    PROCEDURE Draw3D_Block (G,i1,i2,j1,j2,k1,k2 : Integer);
      VAR
        Vert     : ARRAY [1..8,1..2] OF Integer;
        Poly     : ARRAY [1..4,1..2] OF Integer;
        Xsc, Ysc : single;

      PROCEDURE DrawVert (v1, v2 : Integer);
        BEGIN
          Line (Vert[v1,1],Vert[v1,2],Vert[v2,1],Vert[v2,2]);
        END; {DrawVert}

      BEGIN {Draw3D_Block}
        WITH Scale2D DO
          BEGIN
            k1 := k1-1; k2 := k2-1;
            Xsc := Xpx/Xlen; Ysc := Ypx/Ylen;
            Vert[1,1] := Xoff + ROUND (Xsc* (i1*Rot[1,1]+j1*Rot[2,1]+k1*Rot[3,1] -XX1) );
            Vert[1,2] := Yoff + ROUND (Ysc* (i1*Rot[1,2]+j1*Rot[2,2]+k1*Rot[3,2] -YY1) );
            Vert[2,1] := Xoff + ROUND (Xsc* (i2*Rot[1,1]+j1*Rot[2,1]+k1*Rot[3,1] -XX1) );
            Vert[2,2] := Yoff + ROUND (Ysc* (i2*Rot[1,2]+j1*Rot[2,2]+k1*Rot[3,2] -YY1) );
            Vert[3,1] := Xoff + ROUND (Xsc* (i2*Rot[1,1]+j2*Rot[2,1]+k1*Rot[3,1] -XX1) );
            Vert[3,2] := Yoff + ROUND (Ysc* (i2*Rot[1,2]+j2*Rot[2,2]+k1*Rot[3,2] -YY1) );
            Vert[4,1] := Xoff + ROUND (Xsc* (i1*Rot[1,1]+j2*Rot[2,1]+k1*Rot[3,1] -XX1) );
            Vert[4,2] := Yoff + ROUND (Ysc* (i1*Rot[1,2]+j2*Rot[2,2]+k1*Rot[3,2] -YY1) );
            Vert[5,1] := Xoff + ROUND (Xsc* (i1*Rot[1,1]+j1*Rot[2,1]+k2*Rot[3,1] -XX1) );
            Vert[5,2] := Yoff + ROUND (Ysc* (i1*Rot[1,2]+j1*Rot[2,2]+k2*Rot[3,2] -YY1) );
            Vert[6,1] := Xoff + ROUND (Xsc* (i2*Rot[1,1]+j1*Rot[2,1]+k2*Rot[3,1] -XX1) );
            Vert[6,2] := Yoff + ROUND (Ysc* (i2*Rot[1,2]+j1*Rot[2,2]+k2*Rot[3,2] -YY1) );
            Vert[7,1] := Xoff + ROUND (Xsc* (i2*Rot[1,1]+j2*Rot[2,1]+k2*Rot[3,1] -XX1) );
            Vert[7,2] := Yoff + ROUND (Ysc* (i2*Rot[1,2]+j2*Rot[2,2]+k2*Rot[3,2] -YY1) );
            Vert[8,1] := Xoff + ROUND (Xsc* (i1*Rot[1,1]+j2*Rot[2,1]+k2*Rot[3,1] -XX1) );
            Vert[8,2] := Yoff + ROUND (Ysc* (i1*Rot[1,2]+j2*Rot[2,2]+k2*Rot[3,2] -YY1) );
            DrawVert (1,2); DrawVert (1,4); DrawVert (1,5);
            DrawVert (2,3); DrawVert (2,6);
            DrawVert (3,4); DrawVert (3,7);
            DrawVert (4,8);
            DrawVert (5,6); DrawVert (5,8);
            DrawVert (6,7);
            DrawVert (7,8);
          END; {with Scale2D}
        END; {Draw3D_Block}


    PROCEDURE Draw3D_SmAxes;
{      CONST
        h1 = 26;  v1 = 26;
        lsz=20;   csz=sz+7;}
      VAR
        lsz,csz,h1,v1 : Integer;
        h2, v2 : Integer;
      BEGIN
        {Draw little axes}
        lsz:=ROUND(20*SnapWindMx.Sz[1]/400);
        csz:=lsz+7; h1:=csz+1; v1:=csz+1;

        Setcolor (DsCol_Geom); Setcolor (Yellow);
        SetTextStyle (SmallFont, HorizDir, FontSz-2);
        SetTextJustify(CenterText,CenterText);
        h2:= h1+ROUND(lsz*Rot[1,1]); v2:= h1-ROUND(lsz*Rot[1,2]); Line(h1,v1,h2,v2);
        h2:= h1+ROUND(csz*Rot[1,1]); v2:= h1-ROUND(csz*Rot[1,2]); MoveTo(h2,v2); OutText ('X');
        h2:= h1+ROUND(lsz*Rot[2,1]); v2:= h1-ROUND(lsz*Rot[2,2]); Line(h1,v1,h2,v2);
        h2:= h1+ROUND(csz*Rot[2,1]); v2:= h1-ROUND(csz*Rot[2,2]); MoveTo(h2,v2); OutText ('Y');
        h2:= h1+ROUND(lsz*Rot[3,1]); v2:= h1-ROUND(lsz*Rot[3,2]); Line(h1,v1,h2,v2);
        h2:= h1+ROUND(csz*Rot[3,1]); v2:= h1-ROUND(csz*Rot[3,2]); MoveTo(h2,v2); OutText ('Z');
      END; {Draw3D_SmAxes}


     PROCEDURE Ds_Graph2dAxes;
       VAR
         i             : Integer;
         xchar, ychar  : Char;
         lsz,csz,h1,v1,h2,v2 : Integer;
       BEGIN {Ds_Graph2dAxes}
         SetTextStyle (SmallFont, HorizDir, FontSz-1);
         WITH Scale2D DO
          WITH G_CurExt DO
           BEGIN
             IF (Xaxis IN ['t','T'])
                THEN xchar := Xaxis
                ELSE xchar := CHR(ORD(Xaxis)+15);
             IF (Yaxis IN ['t','T'])
                THEN ychar := Yaxis
                ELSE ychar := CHR(ORD(Yaxis)+15);

             Setcolor (DsCol_Fgrnd);
             Ds_DrawWindow (SnapWind);
             ResetViewPort;

             {Write extents}
             IF (Yaxis IN ['j','J'])
                THEN BEGIN
                  WITH SnapWind DO    MoveTo (St[1]-1, St[2]+Sz[2]+1);
                  SetTextJustify (LeftText, TopText);
                  OutText ( '  (' + Xchar + '=' + Strng (G_CurExt.xp1,$F600) +
                            ' , ' + Ychar + '=' + Strng (G_CurExt.yp1,$F600) + ')' );
                  WITH SnapWind DO    MoveTo (St[1]+Sz[1]+1, St[2]-3);
                  SetTextJustify (RightText, BottomText);
                  OutText ( '  (' + Xchar + '=' + Strng (G_CurExt.xp2,$F600) +
                            ' , ' + Ychar + '=' + Strng (G_CurExt.yp2,$F600) + ')' );
                  {draw little axes}
                  lsz:=ROUND(SnapWindMx.Sz[1]/30);
                  h1:=12; v1:=SnapWindMx.St[1]+SnapWindMx.Sz[1]-16; h2:=h1+lsz; v2:=v1-lsz;
                  Setcolor (DsCol_Geom);
                  SetTextStyle (SmallFont, HorizDir, FontSz-2);
                  SetTextJustify(CenterText,CenterText);
                  Line(h1,v1,h2,v1); MoveTo(h2+7,v1); OutText (UpCase(Xchar));
                  Line(h1,v1,h1,v2); MoveTo(h1,v2-7); OutText (UpCase(Ychar));
                END
                ELSE BEGIN
                  WITH SnapWind DO    MoveTo (St[1]-1, St[2]-3);
                  SetTextJustify (LeftText, BottomText);
                  OutText ( '  (' + Xchar + '=' + Strng (G_CurExt.xp1,$F600) +
                            ' , ' + Ychar + '=' + Strng (G_CurExt.yp1,$F600) + ')' );
                  WITH SnapWind DO    MoveTo (St[1]+Sz[1]+1, St[2]+Sz[2]+1);
                  SetTextJustify (RightText, TopText);
                  OutText ( '  (' + Xchar + '=' + Strng (G_CurExt.xp2,$F600) +
                            ' , ' + Ychar + '=' + Strng (G_CurExt.yp2,$F600) + ')' );
                  {draw little axes}
                  lsz:=ROUND(SnapWindMx.Sz[1]/30);
                  h1:=12; v1:=16; h2:=h1+lsz; v2:=v1+lsz;
                  Setcolor (DsCol_Geom);
                  SetTextStyle (SmallFont, HorizDir, FontSz-2);
                  SetTextJustify(CenterText,CenterText);
                  Line(h1,v1,h2,v1); MoveTo(h2+7,v1); OutText (UpCase(Xchar));
                  Line(h1,v1,h1,v2); MoveTo(h1,v2+7); OutText (UpCase(Ychar));
                END;

           END; {With Scale2D}
         ResetLineStyle;
       END; {Ds_Graph2dAxes}


    FUNCTION Inview (Dir, Slice, i1,i2, j1,j2, k1,k2: Integer): Boolean;
      BEGIN
        CASE Dir OF
          0 : Inview := ((Slice>=k1) AND (Slice<=k2));
          1 : Inview := ((Slice>=j1) AND (Slice<=j2));
          2 : Inview := ((Slice>=i1) AND (Slice<=i2));
        END; {Case}
        IF (Slice=0) THEN Inview:=True;
      END; {Inview}


  PROCEDURE Ds_Graph2DGeom (G, slice : Word);
    VAR
      i             : Integer;
      p, otl        : Word;
      ip,jp,kp{,cnt}  : Word;
      cnt           : LongInt;
      ip2,jp2,kp2   : Word;
      dT1,dT2,sol,ipsol,kpsol : Word;
      Mat1          : MatRec;
      GFpos         : LongInt;

    BEGIN {Ds_Graph2DGeom}
      IF (G=0) THEN EXIT;
      Ds_SetWindow (SnapWind);
      Setcolor (DsCol_Geom);
      SetTextStyle (SmallFont, HorizDir, FontSz);
      SetTextJustify (CenterText, CenterText);
{      writeln (tmpdebug, ' end read');
      flush (tmpdebug);}

      WITH Scale2D DO
        BEGIN
          {Materials}

          {Skip if MatOn = 0 .... }

          {Materials : Orig Draw Rectangle - Dashed}
          CASE MatOn OF
            1: WITH Mat1 DO
                BEGIN
                 SetLineType (2);
                 SetColor (LightCyan);
                 GFPos := GeomMap^[G].MatPos;
                 FOR i := 1 TO GeomMap^[G].MattotT DO
                   BEGIN
                     Seek (GeomFile, GFpos);
                     BlockRead (GeomFile, Matnum, 28);
                     CASE Dir OF
                       0 : Draw2D_Rect (i1,j1,i2+1,j2+1);
                       1 : Draw2D_Rect (i1,k1,i2,k2);
                       2 : Draw2D_Rect (k1,j1,k2,j2);
                     END; {Case}
                   GFpos:=GFpos+128;
                   END; {For}
                END; {With}

            2,3: WITH Mat1 DO
                BEGIN
                 SetLineType (2);
                 SetColor (LightCyan);
                 GFPos := GeomMap^[G].MatPos;
                 FOR i := 1 TO GeomMap^[G].MattotT DO
                   BEGIN
                     Seek (GeomFile, GFpos);
                     BlockRead (GeomFile, Matnum, 28);
                     IF (MatOn=2) 
                        THEN SetFillStyle (SolidFill,Matnum)
                        ELSE SetFillStyle (CloseDotFill,Matnum);
                     IF (Matnum>1) THEN
                        CASE Dir OF
                          0 : Draw2D_Bar (i1,j1,i2+1,j2+1);
                          1 : Draw2D_Bar (i1,k1,i2,k2);
                          2 : Draw2D_Bar (k1,j1,k2,j2);
                        END; {Case}
                     GFpos:=GFpos+128;  
                   END; {For}
                END; {With}

          END; {Case}


          {Sources : Draw Rectangle - Filled, or character}
            SetColor (LightCyan);
          SetLineType (5);
{          SetColor (Yellow);
           SetLineType (1);}
          FOR i := 1 TO GeomMap^[G].SourceTot DO
            BEGIN
{              WITH Sources[i] DO
                IF ( (i1=i2) AND (j1=j2) AND (k1=k2) )
                   THEN OutTextXY (Xoff + (Xpx*(i1-XX1)) DIV Xlen, Yoff + (Ypx*(YY2-j1)) DIV Ylen, '*' );}
              WITH Sources[i] DO
               CASE Dir OF
                 0 : Draw2D_Rect (i1,j1,i2,j2);
                 1 : Draw2D_Rect (i1,k1,i2,k2);
                 2 : Draw2D_Rect (k1,j1,k2,j2);
{                 0 : Draw2D_Bar (i1,j1,i2,j2);
                 1 : Draw2D_Bar (i1,k1,i2,k2);
                 2 : Draw2D_Bar (k1,j1,k2,j2);}
               END; {Case}
            END;
          ResetLineStyle;

          {Stopes : Draw Rectangle - Filled}
          FOR i := 1 TO GeomMap^[G].StopeTot DO
           WITH Stopes^[i] DO
            IF (Inview (Dir, Slice, i1,i2, j1,j2, k1,k2)) THEN
              BEGIN
                IF (stype IN [2] )
                   THEN BEGIN  SetColor (LightRed); SetLineStyle (SolidLn,0,Thickwidth);  END
                   ELSE BEGIN  Setcolor (LightRed);SetLineStyle (SolidLn,0,Normwidth);  END;

                IF {(Layout=Nil)} (TRUE)
{                IF ((Layout=Nil) AND (Outline=Nil))}
                   THEN CASE Dir OF
                          0 : Draw2D_Rect (i1,j1,i2,j2);
                          1 : Draw2D_Rect (i1,k1,i2,k2);
                          2 : Draw2D_Rect (k1,j1,k2,j2);
                        END; {Case}

(*                 SetColor (LightRed);
                 SetFillStyle (CloseDotFill,LightRed);
                 Setwritemode(XORPut);
                 cnt := 0;
                 IF (Layout <> Nil) THEN
                    FOR kp := k1 TO k2 DO
                     IF NOT(Keypressed) THEN
                      FOR jp := j1 TO j2 DO
                       IF NOT(Keypressed) THEN
                        FOR ip := i1 TO i2 DO
                          BEGIN
                            inc(cnt);
                            WRITE(tmpdebug,Layout^[cnt]:2);
    {                        IF ((kp<k2) AND (ip<i2)) THEN}
                              IF (Layout^[cnt] = 0) THEN
                                 BEGIN
                                   IF (ip<i2) THEN ip2:=ip+1 ELSE ip2:=ip;
                                   IF (jp<j2) THEN jp2:=jp+1 ELSE jp2:=jp;
                                   IF (kp<k2) THEN kp2:=kp+1 ELSE kp2:=kp;
                                   CASE Dir OF
  {                                   0 : Draw2D_Bar (ip,jp,ip2,jp2);
                                     1 : Draw2D_Bar (ip,kp,ip2,kp2);
                                     2 : Draw2D_Bar (kp,jp,kp2,jp2);}
                                     0 : Draw2D_Rect (ip,jp,ip2,jp2);
                                     1 : Draw2D_Rect (ip,kp,ip2,kp2);
                                     2 : Draw2D_Rect (kp,jp,kp2,jp2);
                                   END; {Case}
                                 END; {If}
                          END; {If, For}
                 Setwritemode(NormalPut);
*)

          {Draw outline only - assume i-k plane}
          dT1:=1; dT2:=i2-i1+1; jp:=j1;
          cnt := 0;
          Setcolor (White); SetLineStyle (SolidLn,0,Normwidth);
          IF (Layout <> Nil) THEN
             FOR kp := k1 TO k2 DO
              IF NOT(Keypressed) THEN
               FOR ip := i1 TO i2 DO
                BEGIN
                  inc(cnt);
                  sol:=Layout^[cnt];
                  IF (ip=i1) THEN ipsol:=1 ELSE ipsol:=Layout^[cnt-dT1];
                  IF (kp=k1) THEN kpsol:=1 ELSE kpsol:=Layout^[cnt-dT2];
                  IF (sol<>ipsol) THEN
                     CASE Dir OF
                       0 : Draw2D_line (ip,jp,ip,jp);
                       1 : Draw2D_line (ip,kp,ip,kp+1);
                       2 : Draw2D_line (kp,jp,kp+1,jp);
                     END; {Case}
                  IF (sol<>kpsol) THEN
                     CASE Dir OF
                       0 : Draw2D_line (ip,jp,ip+1,jp);
                       1 : Draw2D_line (ip,kp,ip+1,kp);
                       2 : Draw2D_line (kp,jp,kp,jp);
                     END; {Case}
                END; {If, For}

(*                 IF (Outline <> Nil)
                    THEN WITH Outline^ DO
                           FOR otl := 1 TO Qty DO
                            FOR p := 2 TO N1 DO
                              CASE Dir OF
                                0 : Draw2D_Line (Pt[p-1,1],j1,Pt[p,1],j2);
                                1 : Draw2D_Line (Pt[p-1,1],Pt[p-1,2],Pt[p,1],Pt[p,2]);
                                2 : Draw2D_Line (Pt[p-1,2],j1,Pt[p,2],j2);
                              END; {Case}
*)

              END; {if/With Stopes^[i]}

{             writeln (tmpdebug, ' stopes - end ');
              flush (tmpdebug);}


          {Draw Model Boundaries}
{          WITH GeomMap^[G] DO
            CASE Dir OF
              0 : Draw2D_Rect (i1,j1,i2,j2);
              1 : Draw2D_Rect (i1,k1,i2,k2);
              2 : Draw2D_Rect (k1,j1,k2,j2);
            END; {Case}

        END; {With Scale2D}

{      DI_ClearGeom (G);}
      ResetLineStyle;
      ResetViewPort;
    END; {Ds_Graph2DGeom}



  PROCEDURE Ds_Graph3DGeom (G : Word);
    VAR
      Xst, Yst, Xfin, Yfin : Integer;
      i                    : Integer;
      Vert                 : ARRAY [1..8,1..2] OF Single;
      p, otl               : Word;
      ip,jp,kp,cnt         : Word;
      ip2,jp2,kp2          : Word;
      dT1,dT2,sol,ipsol,kpsol : Word;

    BEGIN {Ds_Graph3DGeom}
      IF (G=0) THEN EXIT;
      Ds_SetWindow (SnapWindMx);

      Draw3D_SmAxes;

      {Materials : Draw Rectangle - Dashed}
      SetColor (White);
      SetLineType (2);
      FOR i := 1 TO GeomMap^[G].Mattot DO
        WITH Mats[i] DO
          Draw3D_Block (G,i1,i2,j1,j2,k1,k2);

      {Sources : Draw Rectangle - Filled, or character}
      SetColor (Cyan);
      SetLineType (5);
      FOR i := 1 TO GeomMap^[G].SourceTot DO
        WITH Sources[i] DO
          Draw3D_Block (G,i1,i2,j1,j2,k1,k2);

      {Stopes}
      FOR i := 1 TO GeomMap^[G].StopeTot DO
       WITH Stopes^[i] DO
        BEGIN
          CASE stype OF
            0 : BEGIN  SetColor (LightRed);  SetLineType(1);  END;
            1 : BEGIN  SetColor (White);     SetLineType(1);  END;
            2 : BEGIN  SetColor (LightMagenta);  SetLineType(1);  END;
            3 : BEGIN  SetColor (Yellow);  SetLineType(1);  END;
            ELSE BEGIN  SetColor (Yellow); SetLineType(1);  END;
           END; {Case}

{          IF ((NoOutline) AND (Layout=Nil)) THEN}
{          IF (Layout=Nil) THEN}
             Draw3D_Block (G,i1,i2,j1,j2,k1,k2);

(*          SetLineType (1);
          IF ( {(Layout<>Nil) OR }(Outline<>Nil) )
             THEN SetLineStyle (SolidLn,0,Normwidth);
          SetColor (LightRed);
          SetFillStyle (CloseDotFill,LightRed);
          Setwritemode(XORPut);
          cnt := 0;
          IF (Layout <> Nil) THEN
             FOR kp := k1 TO k2 DO
              IF NOT(Keypressed) THEN
               FOR jp := j1 TO j2 DO
                IF NOT(Keypressed) THEN
                 FOR ip := i1 TO i2 DO
                     BEGIN
                       inc(cnt);
                         IF (Layout^[cnt] = 0) THEN
                            BEGIN
                              IF (ip<i2) THEN ip2:=ip+1  ELSE ip2:=ip;
                              IF (jp<j2) THEN jp2:=jp+1  ELSE jp2:=jp;
                              IF (kp<k2) THEN kp2:=kp+1  ELSE kp2:=kp;
                              Draw3D_Block (G,ip,ip2,jp,jp2,kp,kp2);
                            END; {If}
                     END; {If, For}
          Setwritemode(NormalPut);
*)

          {Draw outline only - assume i-k plane}
          dT1:=1; dT2:=i2-i1+1; jp:=j1;
          cnt := 0;
          Setcolor (White); SetLineType(1);
          IF (Layout <> Nil) THEN
             FOR kp := k1 TO k2 DO
              IF NOT(Keypressed) THEN
               FOR ip := i1 TO i2 DO
                BEGIN
                  inc(cnt);
                  sol:=Layout^[cnt];
                  IF (ip=i1) THEN ipsol:=1 ELSE ipsol:=Layout^[cnt-dT1];
                  IF (kp=k1) THEN kpsol:=1 ELSE kpsol:=Layout^[cnt-dT2];
                  IF (sol<>ipsol) THEN Draw3D_line (ip,jp,kp, ip,jp,kp+1);
                  IF (sol<>kpsol) THEN Draw3D_line (ip,jp,kp, ip+1,jp,kp);
                END; {If, For}

{          IF (Outline <> Nil)
             THEN WITH Outline^ DO
                    FOR otl := 1 TO Qty DO
                     FOR p := 2 TO N1 DO
                         Draw3D_Line (Pt[p-1,1],j1,Pt[p-1,2],  Pt[p,1],j2,Pt[p,2]);}
        END; {With Stopes^[i]}

      {Draw Model Boundaries}
{      Setcolor (DsCol_Geom);
      WITH GeomMap^[G] DO
        Draw3D_Block (G,i1,i2,j1,j2,k1,k2);}

{      WITH GeomMap^[G] DO
        FOR jp := j1 TO j2 DO
         FOR kp := k1 TO k2 DO
          FOR ip := i1 TO i2 DO
            BEGIN
              IF (ip<i2) THEN ip2:=ip+1; ELSE ip2:=ip;
              IF (jp<j2) THEN jp2:=jp+1; ELSE jp2:=jp;
              IF (kp<k2) THEN kp2:=kp+1; ELSE kp2:=kp;
              Draw3D_Block (G,ip,ip2,jp,jp2,kp,kp2);
            END; {With, For}

{      DI_ClearGeom (G);}
      ResetLineStyle;
      ResetViewPort;
    END; {Ds_Graph3DGeom}


  PROCEDURE Draw_HistPos;
    VAR
      i    : Integer;
    BEGIN {Draw_HistPos}
      IF (Sets.Set3D)
         THEN Ds_SetWindow (SnapWindMx)
         ELSE Ds_SetWindow (SnapWind);
      Setcolor (DsCol_Geom);
      SetTextStyle (SmallFont, HorizDir, FontSz-2);
      SetTextJustify (CenterText,CenterText);
      FOR i := 1 TO QtyHist DO
        WITH HistMap^[i] DO
          IF (Ax1 IN ['T','t']) THEN
           IF ((i1<>0) AND (j1<>0) AND (i1<>0)) THEN
             IF (Sets.Set3D)
                THEN Draw3D_Point (i1,j1,k1, 'H')
                ELSE
                  CASE Scale2D.Dir OF
                    {0 : Draw2D_Point (i1,j1, 'H');}
                    0 : Draw2D_Point (i1,j1, ' ');
                    1 : Draw2D_Point (i1,k1, ' ');
                    2 : Draw2D_Point (k1,j1, ' ');
                  END; {Case}
    END; {Draw_HistPos}

  PROCEDURE Draw_THistPos;
      {Draws history positions - iff x,y and z included}
    VAR
      i    : Integer;
    BEGIN {Draw_THistPos}
      IF (Traj.NotRead) THEN DI_CalcTraject;
      IF (Sets.Set3D)
         THEN Ds_SetWindow (SnapWindMx)
         ELSE Ds_SetWindow (SnapWind);
      Setcolor (DsCol_Geom);
      SetTextStyle (SmallFont, HorizDir, FontSz-2);
      SetTextJustify (CenterText,CenterText);
      Draw2D_Point (5,5, strng(QtyTraj,$0000));

      FOR i := 1 TO QtyTraj DO
{        WITH HistMap^[i] DO}
{          IF (Ax1 IN ['T','t']) THEN}
        WITH TrajMap.List[i] DO
           IF ((i1<>0) AND (j1<>0) AND (i1<>0)) THEN
             IF (Sets.Set3D)
                THEN Draw3D_Point (i1,j1,k1, strng(i,$0000))
                ELSE
                  CASE Scale2D.Dir OF
                    0 : Draw2D_Point (i1,j1, strng(i,$0000));
                    1 : Draw2D_Point (i1,k1, strng(i,$0000));
                    2 : Draw2D_Point (k1,j1, strng(i,$0000));
                  END; {Case}
    END; {Draw_THistPos}


  PROCEDURE Draw_Traject (Loop:Integer; DelayOn, UnPlot: Boolean);
    CONST
      TrajName:ARRAY [0..2] OF string[12]
        = ('Displ.','Vel.','Accel.');
      TrajUnits:ARRAY [0..2] OF string[3]
        = ('m','m/s','m/s/s');
    VAR
      ih,j,t,tr   : Integer;
      tpos, tnext : Integer;
      tr_sc       : single;
      Vstep       : single;
{      UnPLot  : Boolean;}
    BEGIN {Draw_Traject}
      DelayOn := DelayOn AND (Tdelay>0);
      IF (Keypressed) THEN EXIT;
      IF (Traj.NotRead) THEN DI_CalcTraject;
      IF (Keypressed) THEN EXIT;
      IF (Traj.NotRead) THEN DI_ReadTraject(CurGeom);
      IF (Keypressed) THEN EXIT;
      IF (Sets.VectSet)
         THEN Ds_VectSc := Sets.VectSc
         ELSE IF (Traj.MaxV = 0)
                 THEN Ds_VectSc:=1
                 ELSE Ds_VectSc := 4 * G_CurSc / Traj.MaxV;
      Ds_VectSc := Ds_VectSc*Vzoom;
      tr_sc := Ds_VectSc;
      SetTextJustify (LeftText, CenterText);
{      UnPlot := False;}

      REPEAT
        tpos := 0;
        WITH TrajMap DO
        WITH map  DO
        WITH Traj DO
        FOR t := 1 TO (Np-1) DO
          IF NOT(Keypressed) THEN
            BEGIN
              {First Set Legend}
                Ds_SetWindow (LegWind);
                Vstep := LegWind.Sz[2] * 0.5;
                SetColor (DsCol_Fgrnd);
                MoveTo ( Hinset, TRUNC(Vstep * 0.575)  );
                SetTextStyle (SmallFont, HorizDir, FontSz);
                OutText (TrajName[Traj.Trtype]+' Trajectories');
                MoveTo ( Hinset, TRUNC(Vstep * 0.7)  );
                SetTextStyle (SmallFont, HorizDir, FontSz-1);
                OutText (' Scale= 1:'+strng(1.0/tr_sc,$F301)+TrajUnits[Traj.Trtype]);
                MoveTo ( Hinset, TRUNC(Vstep * 0.750)  );
                IF (Traj.Dscale) THEN OutText ('           / distance');
                MoveTo ( Hinset, TRUNC(Vstep * 0.825)  );
                IF (Traj.Dscale) THEN OutText (' (scaled by distance to ');
                MoveTo ( Hinset, TRUNC(Vstep * 0.875)  );
                IF (Traj.Dscale) THEN OutText ('  source)');
                ResetViewPort;

              IF (UnPlot)
                 THEN SetColor (DsCol_Bgrnd)
                 ELSE SetColor (ColOffset+(t*Sets.QtyCols-1) DIV Np);

              IF (Sets.Set3D)
                 THEN Ds_SetWindow (SnapWindMx)
                 ELSE Ds_SetWindow (SnapWind);
              FOR tr := 1 TO Nt DO
                BEGIN
                  tnext := tpos + Traj.Nt*3;
                  WITH List[tr] DO
                    IF (Sets.Set3D)
                       THEN
                         Draw3D_rLine (i1+Data^[tpos+1]*tr_sc,  j1+Data^[tpos+2]*tr_sc,  k1+Data^[tpos+3]*tr_sc,
                                       i1+Data^[tnext+1]*tr_sc, j1+Data^[tnext+2]*tr_sc, k1+Data^[tnext+3]*tr_sc)
                       ELSE
                         CASE Scale2D.Dir OF
                           0 : Draw2D_rLine (i1+Data^[tpos+1]*tr_sc,  j1+Data^[tpos+2]*tr_sc,
                                             i1+Data^[tnext+1]*tr_sc, j1+Data^[tnext+2]*tr_sc);
                           1 : Draw2D_rLine (i1+Data^[tpos+1]*tr_sc,  k1+Data^[tpos+3]*tr_sc,
                                             i1+Data^[tnext+1]*tr_sc, k1+Data^[tnext+3]*tr_sc);
                           2 : Draw2D_rLine (k1+Data^[tpos+3]*tr_sc,  j1+Data^[tpos+2]*tr_sc,
                                             k1+Data^[tnext+3]*tr_sc, j1+Data^[tnext+2]*tr_sc);
                         END; {Case}
                   tpos := tpos+3;
                END; {For}

              IF (NOT(Keypressed) AND DelayOn)
                 THEN IF (Unplot)
                         THEN Delay(Tdelay DIV 2)
                         ELSE Delay(Tdelay);
            END; {If,For}

        Unplot := NOT(Unplot);
        Dec(Loop);
      UNTIL (Keypressed OR (Loop<=0));

{      Sound (100); Delay(1000); NoSound;}
    END; {Draw_Traject}


  PROCEDURE Ds_GeomLeg (G : Word);
    VAR
      Mid          : Word;
      Vstep        : Single;
    BEGIN
      Ds_SetWindow (LegWind);
      WITH SnapMap^ [G] DO
      WITH LegWind DO
        BEGIN
          Vstep := LegWind.Sz[2] * 0.5;
          Mid   := LegWind.Sz[1] DIV 2;
          SetColor (DsCol_Fgrnd);
          SetTextStyle (SmallFont, HorizDir, FontSz);
          SetTextJustify (CenterText, CenterText);
          MoveTo ( Sz[1] DIV 2, TRUNC(Vstep*0.1) );
          OutText ('FILE = '+Sets.Fname);
          SetTextStyle (SmallFont, HorizDir, FontSz-1);
          MoveTo ( Sz[1] DIV 2, TRUNC(Vstep*0.185) );
          OutText ('('+Sets.Drive+':'+Sets.Dir+'\)');

          SetTextStyle (SmallFont, HorizDir, FontSz);
          SetTextJustify (LeftText, CenterText);
          MoveTo ( Hinset, TRUNC(Vstep * 0.350)  );
          OutText ('  GEOMETRY  [' + Strng (G,0) + ']');
          MoveTo ( Hinset, TRUNC(Vstep * 0.450)  );
          IF (SliceG) THEN OutText ('    Slice (' + Strng (Slice2D,0) + ')');
{         MoveTo ( Hinset, TRUNC(Vstep * 0.15)  );
          OutText (' Time = ' + Strng (Time,$F300) );}
        END; {With}
      ResetViewPort;
    END; {Ds_GeomLeg}


  PROCEDURE Ds_GraphGeom (G : Word);
    BEGIN
      IF (G=0) THEN EXIT;
      WITH GeomMap^ [G] DO
        IF (Sets.Set3D)
           THEN BEGIN
                  Ds_Graph3DGeom (G);
                END
           ELSE BEGIN
                  Ds_Graph2dAxes;
                  Ds_Graph2DGeom (G,Slice2D);
                END;
      {Draw_HistPos;}
      Draw_THistPos;
      Ds_GeomLeg (G);
{      Ds_GeomLeg (CurGeom);}
      ResetViewPort;
      IF (Sets.ShowTitles) THEN Ds_Titles;
    END; {Ds_GraphGeom}


  PROCEDURE Ds_GraphCrackD (D,slice,CrackT : Word;  VAR ch:char);
    CONST
      vopen = 0.25;
    TYPE
      crtyp      = RECORD stid,iqty,i1,i2,j1,j2,k1,k2 : LongInt; END;
      CrackArray = ARRAY [1..800] OF CrackRec;
    VAR
      cr            : crtyp;
      Cracks        : ^CrackArray;
      CrackF        : File;
      CrPos,isz     : LongInt;
      ist, ic       : LongInt;
      i,j,k         : LongInt;
      xp1,xp2,yp1,yp2 : single;
      xp3,xp4,yp3,yp4 : single;
      vfact           : single;
    BEGIN {Ds_GraphCrackD}
      vfact := vzoom*5e3;
      Ds_ClearWindow (SnapWind);
      Ds_SetWindow (SnapWind);
      Setcolor (DsCol_Geom);
      SetTextStyle (SmallFont, HorizDir, FontSz);
      SetTextJustify (CenterText, CenterText);

      CrPos := CrackDMap^[D].Offset;
      ASSIGN (CrackF, (Sets.Drive+':'+Sets.Dir+'\'+Sets.FName+'.CRK') );
      RESET  (CrackF, 1);

      Setcolor (LightRed);SetLineStyle (SolidLn,0,Normwidth);

      FOR ist := 1 TO CrackDMap^[D].StQty DO
        BEGIN
          Seek (CrackF, CrPos+4);
          BlockRead (CrackF,cr.stid,8*4);
          CrPos := CrPos+128;
          isz := cr.iqty*(CrackDMap^[D].Vqty+2);
          GetMem (Cracks,isz*4);
          Seek (CrackF, CrPos);
          BlockRead (CrackF,Cracks^,isz*4);
          CrPos := CrPos+isz*4;

{          IF (Inview (Scale2D.Dir, Slice, i1,i2, j1,j2, k1,k2)) THEN}
          ic:=1;
          WITH cr DO
            BEGIN
             i:=i1; j:=j1;
             IF (j1=j2)
                THEN
                  FOR i:= i1 TO i2 DO
                    BEGIN
                      CASE CrackT OF
                        0: ;
                        1: BEGIN
                             Setcolor (Green);
                             IF (Cracks^[ic].Taslp>0.0) THEN Setcolor (LightBlue);
                             IF (Cracks^[ic].sNs1=0.0) THEN Setcolor (LightRed);
                             IF (i>i1) THEN Draw2D_rLine (i-1,j,i,j);
                           END;
                        2: BEGIN
                             Setcolor (DsCol_Bgrnd);
                             IF (Cracks^[ic].Taslp>0.0) THEN Setcolor (Blue);
                             IF (Cracks^[ic].sNs1=0.0) THEN Setcolor (LightRed);
                             IF (i>i1) THEN Draw2D_rLine (i-1,j,i,j);
                           END;
                        3: BEGIN
                             xp1 := i+Cracks^[ic].uT1s1*vfact;
                             yp1 := (j+vopen)+Cracks^[ic].uNs1*vfact;
                             xp3 := i+Cracks^[ic].uT1s0*vfact;
                             yp3 := (j-vopen)+Cracks^[ic].uNs0*vfact;
                             Setcolor (LightRed); IF (i>i1) THEN Draw2D_rLine (xp1,yp1,xp2,yp2);
                             Setcolor (Blue); IF (i>i1) THEN Draw2D_rLine (xp3,yp3,xp4,yp4);
                             xp2:=xp1; yp2:=yp1;xp4:=xp3; yp4:=yp3;
                           END;
                      END; {Case}
                      inc(ic);
                    END;
             IF (i1=i2)
                THEN
                  FOR j:= j1 TO j2 DO
                    BEGIN
                      CASE CrackT OF
                        0: ;
                        1: BEGIN
                             Setcolor (DsCol_Geom);
                             IF (Cracks^[ic].Taslp>0.0) THEN Setcolor (Blue);
                             IF (Cracks^[ic].sNs1=0.0) THEN Setcolor (LightRed);
                             IF (j>j1) THEN Draw2D_rLine (i,j-1,i,j);
                           END;
                        2: BEGIN
                             Setcolor (DsCol_Bgrnd);
                             IF (Cracks^[ic].Taslp>0.0) THEN Setcolor (Blue);
                             IF (Cracks^[ic].sNs1=0.0) THEN Setcolor (LightRed);
                             IF (j>j1) THEN Draw2D_rLine (i,j-1,i,j);
                           END;
                        3: BEGIN
                             xp1 := (i+vopen)+Cracks^[ic].uNs1*vfact;
                             yp1 := j+Cracks^[ic].uT1s1*vfact;
                             xp3 := (i-vopen)+Cracks^[ic].uNs0*vfact;
                             yp3 := j+Cracks^[ic].uT1s0*vfact;
                             Setcolor (LightRed);  IF (j>j1) THEN Draw2D_rLine (xp1,yp1,xp2,yp2);
                             Setcolor (Blue);      IF (j>j1) THEN Draw2D_rLine (xp3,yp3,xp4,yp4);
                             xp2:=xp1; yp2:=yp1; xp4:=xp3; yp4:=yp3;
                           END;
                      END; {Case}
                      inc(ic);
                    END;

            END; {With}

          FreeMem (Cracks,isz*4);
        END; {With,For}

      CLOSE (CrackF);

      IF (Sstep) THEN ch := ReadKey;
      IF NOT(Sstep) THEN WHILE (Keypressed) DO ch:=Readkey;
      IF (ch=' ') THEN Sstep:=NOT(Sstep);
      IF NOT(ch IN ['G','g','X','x',#27]) THEN ch := '~';
    END; {Ds_GraphCrackD}

